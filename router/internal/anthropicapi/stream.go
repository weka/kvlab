package anthropicapi

import (
	"bytes"
	"encoding/json"
	"io"
	"time"

	"github.com/weka/kvlab/router/internal/clock"
)

// streamConverter turns a Chat Completions SSE stream into an Anthropic one.
//
// Anthropic streams content blocks strictly one after another — start, deltas,
// stop — each with its own index; Chat Completions interleaves freely (text,
// reasoning and several tool calls, each keyed by its own index). The
// converter serialises: one block is open at a time, and a tool call whose
// deltas arrive while another block is open is buffered and emitted whole once
// it is its turn. In practice vLLM finishes one call's arguments before the
// next starts, so only the first call's arguments ever wait, and only for the
// length of the next.
type streamConverter struct {
	w     io.Writer
	flush func()
	tr    *translated
	clk   clock.Clock
	ping  time.Duration
	warn  func(msg string, args ...any)

	line bytes.Buffer // partial upstream line across Write calls
	data bytes.Buffer // data: lines of the current upstream event

	events   int // events written so far
	started  bool
	finished bool // message_stop or error sent; nothing more may follow
	lastEmit time.Time

	id    string
	block blockKind // the open block, if any
	index int       // its Anthropic index
	next  int       // next Anthropic index to assign

	tools  []*toolState // in order of first appearance
	byIdx  map[int]*toolState
	cur    *toolState // the tool whose block is open
	finish string
	stop   json.RawMessage
	usage  *oaUsage
	sawAny bool // any content at all — for the empty-response case
}

type blockKind int

const (
	blockNone blockKind = iota
	blockText
	blockThinking
	blockTool
)

type toolState struct {
	id     string
	name   string
	args   bytes.Buffer
	opened bool
	closed bool
}

func newStreamConverter(w io.Writer, flush func(), tr *translated, clk clock.Clock,
	ping time.Duration, warn func(string, ...any)) *streamConverter {
	return &streamConverter{w: w, flush: flush, tr: tr, clk: clk, ping: ping, warn: warn,
		byIdx: map[int]*toolState{}}
}

// event writes one Anthropic SSE event.
func (c *streamConverter) event(typ string, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	var buf bytes.Buffer
	buf.Grow(len(typ) + len(b) + 16)
	buf.WriteString("event: ")
	buf.WriteString(typ)
	buf.WriteString("\ndata: ")
	buf.Write(b)
	buf.WriteString("\n\n")
	_, _ = c.w.Write(buf.Bytes())
	c.events++
	c.lastEmit = c.clk.Now()
}

// start sends message_start and the first ping, as Anthropic does.
func (c *streamConverter) start(upstreamID string, estInput int) {
	if c.started {
		return
	}
	c.started = true
	if c.id == "" {
		c.id = messageID(upstreamID)
	}
	c.event("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": c.id, "type": "message", "role": "assistant",
			"model": c.tr.clientModel, "content": []any{},
			"stop_reason": nil, "stop_sequence": nil,
			"usage": Usage{InputTokens: estInput, OutputTokens: 1},
		},
	})
	c.event("ping", map[string]string{"type": "ping"})
	c.flush()
}

// Write consumes upstream SSE bytes in arbitrary chunks.
func (c *streamConverter) Write(p []byte) (int, error) {
	n := len(p)
	before := c.events
	for len(p) > 0 {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			c.line.Write(p)
			break
		}
		c.line.Write(p[:i])
		p = p[i+1:]
		line := bytes.TrimSuffix(c.line.Bytes(), []byte("\r"))
		c.handleLine(line)
		c.line.Reset()
	}
	if c.events == before && c.started && !c.finished && c.clk.Now().Sub(c.lastEmit) >= c.ping {
		// The backend is producing (reasoning that is not surfaced, say) but
		// the client has seen nothing for a while.
		c.event("ping", map[string]string{"type": "ping"})
	}
	if c.events != before {
		c.flush()
	}
	return n, nil
}

// handleLine processes one SSE line.
func (c *streamConverter) handleLine(line []byte) {
	switch {
	case len(line) == 0:
		if c.data.Len() == 0 {
			return
		}
		payload := append([]byte(nil), c.data.Bytes()...)
		c.data.Reset()
		c.handleData(payload)
	case line[0] == ':':
		return // comment
	case bytes.HasPrefix(line, []byte("data:")):
		d := bytes.TrimPrefix(line, []byte("data:"))
		d = bytes.TrimPrefix(d, []byte(" "))
		if c.data.Len() > 0 {
			c.data.WriteByte('\n')
		}
		c.data.Write(d)
	}
}

func (c *streamConverter) handleData(d []byte) {
	if c.finished {
		return
	}
	if bytes.Equal(bytes.TrimSpace(d), []byte("[DONE]")) {
		c.complete()
		return
	}
	var ch oaResponse
	if err := json.Unmarshal(d, &ch); err != nil {
		c.warn("unparseable chunk from backend", "chunk", truncate(string(d), 300))
		return
	}
	if ch.Error != nil {
		c.fail("api_error", "backend error: "+ch.Error.Message)
		return
	}
	c.start(ch.ID, 0)
	if ch.Usage != nil {
		c.usage = ch.Usage
	}
	for _, choice := range ch.Choices {
		if choice.Index != 0 {
			continue // n>1 is never requested
		}
		if d := choice.Delta; d != nil {
			if r := d.reasoning(); r != "" && c.tr.surfaceThinking {
				c.text(blockThinking, r)
			}
			if d.Content != nil && *d.Content != "" {
				c.text(blockText, *d.Content)
			}
			for _, tc := range d.ToolCalls {
				c.toolDelta(tc)
			}
		}
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			c.finish = *choice.FinishReason
			if len(choice.StopReason) > 0 && string(choice.StopReason) != "null" {
				c.stop = choice.StopReason
			}
		}
	}
}

func (c *streamConverter) openBlock(kind blockKind, cb map[string]any) {
	c.closeBlock()
	c.block = kind
	c.index = c.next
	c.next++
	c.sawAny = true
	c.event("content_block_start", map[string]any{
		"type": "content_block_start", "index": c.index, "content_block": cb})
}

func (c *streamConverter) closeBlock() {
	if c.block == blockNone {
		return
	}
	if c.block == blockThinking {
		// Anthropic closes every thinking block with its signature. There is
		// no real one to give; an empty signature tells a client there is
		// nothing to verify, and history sent back here drops it anyway.
		c.event("content_block_delta", map[string]any{"type": "content_block_delta",
			"index": c.index, "delta": map[string]string{"type": "signature_delta", "signature": ""}})
	}
	if c.block == blockTool && c.cur != nil {
		if c.cur.args.Len() == 0 {
			// A call with no arguments still has an input: {}.
			c.event("content_block_delta", map[string]any{"type": "content_block_delta",
				"index": c.index, "delta": map[string]string{"type": "input_json_delta", "partial_json": "{}"}})
		} else if _, ok := toolInput(c.cur.args.String()); !ok {
			c.warn("tool call arguments are not a JSON object", "tool", c.cur.name,
				"arguments", truncate(c.cur.args.String(), 300))
		}
		c.cur.closed = true
		c.cur = nil
	}
	c.event("content_block_stop", map[string]any{"type": "content_block_stop", "index": c.index})
	c.block = blockNone
}

func (c *streamConverter) text(kind blockKind, s string) {
	if c.block != kind {
		if kind == blockText {
			c.openBlock(kind, map[string]any{"type": "text", "text": ""})
		} else {
			c.openBlock(kind, map[string]any{"type": "thinking", "thinking": "", "signature": ""})
		}
	}
	if kind == blockText {
		c.event("content_block_delta", map[string]any{"type": "content_block_delta",
			"index": c.index, "delta": map[string]string{"type": "text_delta", "text": s}})
	} else {
		c.event("content_block_delta", map[string]any{"type": "content_block_delta",
			"index": c.index, "delta": map[string]string{"type": "thinking_delta", "thinking": s}})
	}
}

func (c *streamConverter) toolDelta(tc oaToolCall) {
	idx := 0
	if tc.Index != nil {
		idx = *tc.Index
	}
	t := c.byIdx[idx]
	if t == nil {
		t = &toolState{}
		c.byIdx[idx] = t
		c.tools = append(c.tools, t)
	}
	if tc.ID != "" && t.id == "" {
		t.id = tc.ID
	}
	if tc.Function.Name != "" && !t.opened {
		t.name += tc.Function.Name
	}
	if t.closed {
		// Its block is already stopped; Anthropic has no way to reopen one.
		c.warn("tool call arguments arrived after the call's block closed; dropped",
			"tool", t.name, "fragment", truncate(tc.Function.Arguments, 200))
		return
	}
	t.args.WriteString(tc.Function.Arguments)
	if t == c.cur {
		if tc.Function.Arguments != "" {
			c.argsDelta(tc.Function.Arguments)
		}
	}
	c.advance(false)
}

func (c *streamConverter) argsDelta(s string) {
	c.event("content_block_delta", map[string]any{"type": "content_block_delta",
		"index": c.index, "delta": map[string]string{"type": "input_json_delta", "partial_json": s}})
}

// advance opens the next tool block when it may: no tool is open, or the open
// one's arguments are already a complete JSON value (so it is done) and a
// later call has begun. At the end of the stream (final) every pending call is
// emitted regardless.
func (c *streamConverter) advance(final bool) {
	for {
		var next *toolState
		for _, t := range c.tools {
			if !t.opened {
				next = t
				break
			}
		}
		if next == nil {
			return
		}
		if c.cur != nil {
			if !final && !json.Valid(bytes.TrimSpace(c.cur.args.Bytes())) {
				return
			}
			c.closeBlock()
		}
		if next.name == "" && !final {
			return // cannot start a tool_use block without its name
		}
		c.openBlock(blockTool, map[string]any{"type": "tool_use",
			"id": toolUseID(next.id), "name": next.name, "input": map[string]any{}})
		next.opened = true
		c.cur = next
		if next.args.Len() > 0 {
			c.argsDelta(next.args.String())
		}
		// Loop: if this call's buffered arguments are already complete and a
		// later call is waiting, it can follow at once.
	}
}

// complete ends a stream the backend finished.
func (c *streamConverter) complete() {
	if c.finished {
		return
	}
	if c.finish == "" {
		c.fail("api_error", "backend stream ended without a finish reason")
		return
	}
	c.start("", 0)
	c.advance(true)
	c.closeBlock()
	if !c.sawAny {
		// A message with no content is legal, but a client that renders the
		// last text block gets nothing at all; an empty text block is what
		// Anthropic itself returns for an empty generation.
		c.openBlock(blockText, map[string]any{"type": "text", "text": ""})
		c.closeBlock()
	}
	sr, seq := stopReason(c.finish, c.stop, c.tr.stopSequences, len(c.tools) > 0)
	c.event("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": sr, "stop_sequence": seq},
		"usage": c.usage.anthropic(),
	})
	c.event("message_stop", map[string]string{"type": "message_stop"})
	c.finished = true
}

// fail ends the stream with an error event. Anthropic sends these mid-stream
// (overloaded_error is the common one) and clients surface them as errors.
func (c *streamConverter) fail(typ, msg string) {
	if c.finished {
		return
	}
	c.event("error", map[string]any{"type": "error",
		"error": map[string]string{"type": typ, "message": msg}})
	c.finished = true
}

// Close is called when the upstream body has ended. A stream that delivered a
// finish_reason but no [DONE] is still complete; one that delivered neither
// was cut off.
func (c *streamConverter) Close(clientGone bool) {
	if c.line.Len() > 0 {
		c.handleLine(c.line.Bytes())
		c.line.Reset()
	}
	c.handleLine(nil)
	if c.finished || clientGone {
		return
	}
	if c.finish != "" {
		c.complete()
	} else {
		c.fail("api_error", "backend stream ended before the response was complete")
	}
	c.flush()
}
