// Package anthropicsse parses an Anthropic Messages event stream and checks it
// against the protocol's grammar, the way a strict client would consume it.
//
// Test-only. It is the oracle for the router's translation mode: a stream that
// Assemble accepts has the event order, block indexing and delta types an
// Anthropic SDK requires, and Assemble's result is the message such a client
// would end up holding.
package anthropicsse

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// Event is one SSE event: its event: name and its data: payload.
type Event struct {
	Name string
	Data json.RawMessage
}

// Type is the payload's own type field, which must equal Name.
func (e Event) Type() string {
	var v struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(e.Data, &v)
	return v.Type
}

// Parse reads every event from an SSE body.
func Parse(r io.Reader) ([]Event, error) {
	var out []Event
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	var name string
	var data bytes.Buffer
	flush := func() {
		if name == "" && data.Len() == 0 {
			return
		}
		out = append(out, Event{Name: name, Data: append(json.RawMessage(nil), data.Bytes()...)})
		name = ""
		data.Reset()
	}
	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			flush()
		case strings.HasPrefix(line, "event:"):
			name = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		}
	}
	flush()
	return out, sc.Err()
}

// Block is an assembled content block.
type Block struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
}

// Message is what a client holds after the stream.
type Message struct {
	ID           string
	Model        string
	Content      []Block
	StopReason   string
	StopSequence string
	Usage        map[string]int
	// Pings counts ping events, which may appear anywhere after message_start.
	Pings int
	// Error is set when the stream ended with an error event.
	Error *ErrorBody
}

// ErrorBody is an error event's payload.
type ErrorBody struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// Assemble validates the event sequence and returns the resulting message.
//
// The grammar: message_start, then content blocks strictly one at a time
// (content_block_start with the next index, deltas of the block's own kind
// carrying that index, content_block_stop), then message_delta and
// message_stop. ping may appear anywhere after message_start. An error event
// ends the stream early and is reported in Message.Error, not as an error.
func Assemble(events []Event) (*Message, error) {
	if len(events) == 0 {
		return nil, fmt.Errorf("empty stream")
	}
	m := &Message{Usage: map[string]int{}}
	open := -1
	var cur *Block
	var partial strings.Builder
	state := "start" // start -> blocks -> delta -> stopped
	for i, ev := range events {
		if ev.Name != ev.Type() {
			return nil, fmt.Errorf("event %d: name %q but payload type %q", i, ev.Name, ev.Type())
		}
		if ev.Name == "error" {
			var e struct {
				Error ErrorBody `json:"error"`
			}
			if err := json.Unmarshal(ev.Data, &e); err != nil {
				return nil, fmt.Errorf("event %d: error payload: %v", i, err)
			}
			m.Error = &e.Error
			if i != len(events)-1 {
				return nil, fmt.Errorf("event %d: events after an error", i)
			}
			return m, nil
		}
		if state == "start" {
			if ev.Name != "message_start" {
				return nil, fmt.Errorf("event %d: first event is %q, want message_start", i, ev.Name)
			}
			var s struct {
				Message struct {
					ID      string            `json:"id"`
					Type    string            `json:"type"`
					Role    string            `json:"role"`
					Model   string            `json:"model"`
					Content []json.RawMessage `json:"content"`
					Usage   map[string]int    `json:"usage"`
				} `json:"message"`
			}
			if err := json.Unmarshal(ev.Data, &s); err != nil {
				return nil, fmt.Errorf("message_start: %v", err)
			}
			if s.Message.Type != "message" || s.Message.Role != "assistant" || s.Message.ID == "" {
				return nil, fmt.Errorf("message_start: bad message header %s", ev.Data)
			}
			if s.Message.Content == nil || len(s.Message.Content) != 0 {
				return nil, fmt.Errorf("message_start: content must be an empty array")
			}
			if _, ok := s.Message.Usage["input_tokens"]; !ok {
				return nil, fmt.Errorf("message_start: usage.input_tokens missing")
			}
			m.ID, m.Model = s.Message.ID, s.Message.Model
			state = "blocks"
			continue
		}
		if ev.Name == "ping" {
			m.Pings++
			continue
		}
		switch ev.Name {
		case "content_block_start":
			if state != "blocks" || open >= 0 {
				return nil, fmt.Errorf("event %d: content_block_start while a block is open or after message_delta", i)
			}
			var s struct {
				Index        *int  `json:"index"`
				ContentBlock Block `json:"content_block"`
			}
			if err := json.Unmarshal(ev.Data, &s); err != nil {
				return nil, err
			}
			if s.Index == nil || *s.Index != len(m.Content) {
				return nil, fmt.Errorf("event %d: block index %v, want %d", i, s.Index, len(m.Content))
			}
			b := s.ContentBlock
			switch b.Type {
			case "text", "thinking":
			case "tool_use":
				if b.ID == "" || b.Name == "" {
					return nil, fmt.Errorf("event %d: tool_use block without id or name", i)
				}
				if string(b.Input) != "{}" {
					return nil, fmt.Errorf("event %d: tool_use start input %s, want {}", i, b.Input)
				}
				b.Input = nil
			default:
				return nil, fmt.Errorf("event %d: unknown block type %q", i, b.Type)
			}
			m.Content = append(m.Content, b)
			open = *s.Index
			cur = &m.Content[open]
			partial.Reset()
		case "content_block_delta":
			var s struct {
				Index *int `json:"index"`
				Delta struct {
					Type        string  `json:"type"`
					Text        *string `json:"text"`
					Thinking    *string `json:"thinking"`
					Signature   *string `json:"signature"`
					PartialJSON *string `json:"partial_json"`
				} `json:"delta"`
			}
			if err := json.Unmarshal(ev.Data, &s); err != nil {
				return nil, err
			}
			if open < 0 || s.Index == nil || *s.Index != open {
				return nil, fmt.Errorf("event %d: delta for index %v but open block is %d", i, s.Index, open)
			}
			d := s.Delta
			switch {
			case d.Type == "text_delta" && cur.Type == "text" && d.Text != nil:
				cur.Text += *d.Text
			case d.Type == "thinking_delta" && cur.Type == "thinking" && d.Thinking != nil:
				cur.Thinking += *d.Thinking
			case d.Type == "signature_delta" && cur.Type == "thinking" && d.Signature != nil:
				cur.Signature += *d.Signature
			case d.Type == "input_json_delta" && cur.Type == "tool_use" && d.PartialJSON != nil:
				partial.WriteString(*d.PartialJSON)
			default:
				return nil, fmt.Errorf("event %d: delta %s does not fit a %s block", i, ev.Data, cur.Type)
			}
		case "content_block_stop":
			var s struct {
				Index *int `json:"index"`
			}
			_ = json.Unmarshal(ev.Data, &s)
			if open < 0 || s.Index == nil || *s.Index != open {
				return nil, fmt.Errorf("event %d: stop for index %v but open block is %d", i, s.Index, open)
			}
			if cur.Type == "tool_use" {
				js := partial.String()
				if js == "" {
					js = "{}"
				}
				var obj map[string]any
				if err := json.Unmarshal([]byte(js), &obj); err != nil {
					return nil, fmt.Errorf("event %d: tool_use input %q is not a JSON object: %v", i, js, err)
				}
				cur.Input = json.RawMessage(js)
			}
			open = -1
			cur = nil
		case "message_delta":
			if state != "blocks" || open >= 0 {
				return nil, fmt.Errorf("event %d: message_delta with a block open or twice", i)
			}
			var s struct {
				Delta struct {
					StopReason   *string `json:"stop_reason"`
					StopSequence *string `json:"stop_sequence"`
				} `json:"delta"`
				Usage map[string]int `json:"usage"`
			}
			if err := json.Unmarshal(ev.Data, &s); err != nil {
				return nil, err
			}
			if s.Delta.StopReason == nil {
				return nil, fmt.Errorf("event %d: message_delta without stop_reason", i)
			}
			if _, ok := s.Usage["output_tokens"]; !ok {
				return nil, fmt.Errorf("event %d: message_delta usage.output_tokens missing", i)
			}
			m.StopReason = *s.Delta.StopReason
			if s.Delta.StopSequence != nil {
				m.StopSequence = *s.Delta.StopSequence
			}
			m.Usage = s.Usage
			state = "delta"
		case "message_stop":
			if state != "delta" {
				return nil, fmt.Errorf("event %d: message_stop before message_delta", i)
			}
			if i != len(events)-1 {
				return nil, fmt.Errorf("event %d: events after message_stop", i)
			}
			state = "stopped"
		default:
			return nil, fmt.Errorf("event %d: unknown event %q", i, ev.Name)
		}
	}
	if state != "stopped" {
		return nil, fmt.Errorf("stream ended in state %q without message_stop", state)
	}
	return m, nil
}
