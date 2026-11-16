package mockvllm

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/weka/kvlab/kvcache"
)

// ScriptHeader carries a scripted reply on a chat request. It is how a test
// makes the mock answer with reasoning, tool calls, a particular finish reason
// or a broken stream, without the mock knowing anything about the test.
//
// A request without it gets the synthetic-token reply every benchmark relies
// on, unless the server runs agentic (Config.Agentic) and the request offers
// tools — see agenticScript.
const ScriptHeader = "X-Mock-Script"

// Script is a scripted chat completion.
type Script struct {
	// Reasoning is streamed before anything else, under ReasoningField
	// ("reasoning_content", the default, or "reasoning" — vLLM has used both).
	Reasoning      string           `json:"reasoning,omitempty"`
	ReasoningField string           `json:"reasoning_field,omitempty"`
	Text           string           `json:"text,omitempty"`
	ToolCalls      []ScriptToolCall `json:"tool_calls,omitempty"`
	// FinishReason defaults to "tool_calls" when there are tool calls and to
	// "stop" otherwise.
	FinishReason string `json:"finish_reason,omitempty"`
	// StopReason is vLLM's stop_reason: the stop string that ended the reply.
	StopReason string `json:"stop_reason,omitempty"`
	// ChunkChars splits every streamed string into pieces this long, so a
	// client's reassembly across chunk boundaries is exercised. Default 3.
	ChunkChars int `json:"chunk_chars,omitempty"`
	// InterleaveTools streams parallel calls' argument pieces round-robin
	// rather than one call after another.
	InterleaveTools bool `json:"interleave_tools,omitempty"`
	// CutAfterChunks ends the stream after this many content chunks with no
	// finish_reason and no [DONE], as a backend that died mid-reply would.
	CutAfterChunks int `json:"cut_after_chunks,omitempty"`
	// ErrorStatus answers with this HTTP status and ErrorMessage instead.
	ErrorStatus  int    `json:"error_status,omitempty"`
	ErrorMessage string `json:"error_message,omitempty"`
	// ChunkDelay paces the stream, overriding the engine's output rate.
	ChunkDelay time.Duration `json:"chunk_delay,omitempty"`
}

// ScriptToolCall is one call in a scripted reply. Arguments is the raw JSON
// text, streamed in ChunkChars pieces.
type ScriptToolCall struct {
	ID        string `json:"id,omitempty"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func (sc *Script) finish() string {
	if sc.FinishReason != "" {
		return sc.FinishReason
	}
	if len(sc.ToolCalls) > 0 {
		return "tool_calls"
	}
	return "stop"
}

func (sc *Script) outputTokens() int {
	n := len(sc.Reasoning) + len(sc.Text)
	for _, tc := range sc.ToolCalls {
		n += len(tc.Name) + len(tc.Arguments)
	}
	return max(1, n/4)
}

func (sc *Script) pieces(s string) []string {
	n := sc.ChunkChars
	if n <= 0 {
		n = 3
	}
	var out []string
	for len(s) > 0 {
		k := min(n, len(s))
		out = append(out, s[:k])
		s = s[k:]
	}
	return out
}

// scriptFor returns the reply script for a request, or nil for the default
// synthetic reply. A malformed header is answered as a 400 by the caller.
func (s *Server) scriptFor(r *http.Request, req *chatCompletionRequest) (*Script, error) {
	if h := r.Header.Get(ScriptHeader); h != "" {
		var sc Script
		if err := json.Unmarshal([]byte(h), &sc); err != nil {
			return nil, err
		}
		return &sc, nil
	}
	if s.engine.Config().Agentic {
		return agenticScript(req), nil
	}
	return nil, nil
}

// agenticScript plays a minimal tool-using assistant, so an agent client (an
// Anthropic client behind the router's translation mode, say) can run whole
// multi-turn loops against the mock: offered tools and not just given a tool
// result, it calls the first tool with arguments built from its schema;
// otherwise it answers in text. Returns nil — the synthetic reply — for a
// request without tools, so benchmark traffic is unaffected.
func agenticScript(req *chatCompletionRequest) *Script {
	var tools []struct {
		Function struct {
			Name       string          `json:"name"`
			Parameters json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	if len(req.Tools) == 0 || json.Unmarshal(req.Tools, &tools) != nil || len(tools) == 0 {
		return nil
	}
	sc := &Script{}
	if req.ChatTemplateKwargs["thinking"] == true || req.ChatTemplateKwargs["enable_thinking"] == true {
		sc.Reasoning = "The user needs an answer; let me work it out."
	}
	if n := len(req.Messages); n > 0 && req.Messages[n-1].Role == "tool" {
		sc.Text = "Done: the tool returned " + strconv.Itoa(len(extractText(req.Messages[n-1].Content))) + " characters."
		return sc
	}
	t := tools[0].Function
	sc.ToolCalls = []ScriptToolCall{{Name: t.Name, Arguments: argsFromSchema(t.Parameters)}}
	return sc
}

// argsFromSchema fills a schema's required properties with placeholder values
// of the right type.
func argsFromSchema(raw json.RawMessage) string {
	var schema struct {
		Properties map[string]struct {
			Type string `json:"type"`
		} `json:"properties"`
		Required []string `json:"required"`
	}
	_ = json.Unmarshal(raw, &schema)
	args := map[string]any{}
	for _, name := range schema.Required {
		switch schema.Properties[name].Type {
		case "integer", "number":
			args[name] = 1
		case "boolean":
			args[name] = true
		case "array":
			args[name] = []any{}
		case "object":
			args[name] = map[string]any{}
		default:
			args[name] = "mock"
		}
	}
	b, _ := json.Marshal(args)
	return string(b)
}

// scriptMsg is the message/delta shape a scripted reply needs, which the
// synthetic path's chatMsgOut deliberately does not carry.
type scriptMsg struct {
	Role             string           `json:"role,omitempty"`
	Content          *string          `json:"content,omitempty"`
	ReasoningContent *string          `json:"reasoning_content,omitempty"`
	Reasoning        *string          `json:"reasoning,omitempty"`
	ToolCalls        []scriptToolCall `json:"tool_calls,omitempty"`
}

type scriptToolCall struct {
	Index    int                `json:"index"`
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function scriptFunctionCall `json:"function"`
}

type scriptFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

type scriptChoice struct {
	Index        int        `json:"index"`
	Message      *scriptMsg `json:"message,omitempty"`
	Delta        *scriptMsg `json:"delta,omitempty"`
	FinishReason *string    `json:"finish_reason"`
	StopReason   *string    `json:"stop_reason"`
}

type scriptResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []scriptChoice `json:"choices"`
	Usage   *usage         `json:"usage,omitempty"`
}

func (sc *Script) reasoningMsg(s string) *scriptMsg {
	if sc.ReasoningField == "reasoning" {
		return &scriptMsg{Reasoning: &s}
	}
	return &scriptMsg{ReasoningContent: &s}
}

func (sc *Script) callID(i int) string {
	if id := sc.ToolCalls[i].ID; id != "" {
		return id
	}
	return "call_mock_" + strconv.Itoa(i)
}

// serveScript answers a chat request from a script. Admission, prefill timing
// and cache accounting are the caller's, exactly as for the synthetic reply.
func (s *Server) serveScript(w http.ResponseWriter, r *http.Request, req chatCompletionRequest,
	sc *Script, units []kvcache.Unit, res Residency, work time.Duration) {
	ctx := r.Context()
	modelID := s.engine.Config().ModelID
	id := s.newID("chatcmpl")
	//clockexempt: cosmetic OpenAI wire-format timestamp, not a routing or timing decision
	created := time.Now().Unix()
	outTok := sc.outputTokens()
	finish := sc.finish()
	var stop *string
	if sc.StopReason != "" {
		stop = &sc.StopReason
	}

	if !req.Stream {
		if !s.engine.AwaitTTFT(ctx, work) {
			return
		}
		msg := &scriptMsg{Role: "assistant"}
		if sc.Text != "" || len(sc.ToolCalls) == 0 {
			msg.Content = &sc.Text
		}
		if sc.Reasoning != "" {
			rm := sc.reasoningMsg(sc.Reasoning)
			msg.ReasoningContent, msg.Reasoning = rm.ReasoningContent, rm.Reasoning
		}
		for i, tc := range sc.ToolCalls {
			msg.ToolCalls = append(msg.ToolCalls, scriptToolCall{Index: i, ID: sc.callID(i),
				Type: "function", Function: scriptFunctionCall{Name: tc.Name, Arguments: tc.Arguments}})
		}
		u := buildUsage(res.Total, res.Cached(), outTok)
		writeJSON(w, http.StatusOK, scriptResponse{ID: id, Object: "chat.completion",
			Created: created, Model: modelID, Usage: &u,
			Choices: []scriptChoice{{Message: msg, FinishReason: &finish, StopReason: stop}}})
		s.finishScript(units, res, outTok)
		return
	}

	flusher := startSSE(w)
	if !s.engine.AwaitTTFT(ctx, work) {
		return
	}
	delay := sc.ChunkDelay
	if delay <= 0 {
		delay = s.engine.OutputTokenInterval()
	}
	chunk := func(d *scriptMsg) scriptResponse {
		return scriptResponse{ID: id, Object: "chat.completion.chunk", Created: created,
			Model: modelID, Choices: []scriptChoice{{Delta: d}}}
	}
	// Every delta of the reply, in order, before pacing.
	deltas := []*scriptMsg{{Role: "assistant"}}
	for _, p := range sc.pieces(sc.Reasoning) {
		deltas = append(deltas, sc.reasoningMsg(p))
	}
	for _, p := range sc.pieces(sc.Text) {
		deltas = append(deltas, &scriptMsg{Content: &p})
	}
	deltas = append(deltas, sc.toolDeltas()...)

	sent := 0
	for i, d := range deltas {
		if i > 0 && !sleepCtx(ctx, delay) {
			return // client gone
		}
		if sc.CutAfterChunks > 0 && i >= sc.CutAfterChunks {
			return // a backend dying mid-reply: no finish, no [DONE]
		}
		writeSSE(w, flusher, chunk(d))
		sent++
	}
	final := chunk(&scriptMsg{})
	final.Choices[0].FinishReason = &finish
	final.Choices[0].StopReason = stop
	writeSSE(w, flusher, final)
	if req.wantsUsage() {
		// vLLM sends usage as its own chunk with no choices.
		u := buildUsage(res.Total, res.Cached(), outTok)
		writeSSE(w, flusher, scriptResponse{ID: id, Object: "chat.completion.chunk",
			Created: created, Model: modelID, Choices: []scriptChoice{}, Usage: &u})
	}
	writeSSEDone(w, flusher)
	s.finishScript(units, res, outTok)
}

// toolDeltas streams each call as vLLM does: a head chunk with id and name,
// then the arguments in pieces.
func (sc *Script) toolDeltas() []*scriptMsg {
	heads := make([]*scriptMsg, len(sc.ToolCalls))
	args := make([][]string, len(sc.ToolCalls))
	for i, tc := range sc.ToolCalls {
		heads[i] = &scriptMsg{ToolCalls: []scriptToolCall{{Index: i, ID: sc.callID(i), Type: "function",
			Function: scriptFunctionCall{Name: tc.Name}}}}
		args[i] = sc.pieces(tc.Arguments)
	}
	argDelta := func(i int, p string) *scriptMsg {
		return &scriptMsg{ToolCalls: []scriptToolCall{{Index: i, Function: scriptFunctionCall{Arguments: p}}}}
	}
	var out []*scriptMsg
	if !sc.InterleaveTools {
		for i := range sc.ToolCalls {
			out = append(out, heads[i])
			for _, p := range args[i] {
				out = append(out, argDelta(i, p))
			}
		}
		return out
	}
	out = append(out, heads...)
	for k := 0; ; k++ {
		more := false
		for i := range args {
			if k < len(args[i]) {
				out = append(out, argDelta(i, args[i][k]))
				more = true
			}
		}
		if !more {
			return out
		}
	}
}

func (s *Server) finishScript(units []kvcache.Unit, res Residency, outTok int) {
	s.engine.RecordOutput(outTok)
	s.engine.AppendOutputBlocks(units, outTok, strings.Repeat("x", outTok))
	s.engine.PublishToTier(units)
	s.coll.observe("success", res, outTok)
}
