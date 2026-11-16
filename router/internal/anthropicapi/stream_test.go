package anthropicapi

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/clock"
	"github.com/weka/kvlab/router/internal/testutil/anthropicsse"
)

// upstream renders Chat Completions chunks as vLLM streams them.
func upstream(done bool, chunks ...string) string {
	var sb strings.Builder
	for _, c := range chunks {
		sb.WriteString("data: " + c + "\n\n")
	}
	if done {
		sb.WriteString("data: [DONE]\n\n")
	}
	return sb.String()
}

func delta(d string) string {
	return `{"id":"chatcmpl-s1","object":"chat.completion.chunk","choices":[{"index":0,"delta":` + d + `,"finish_reason":null}]}`
}

func finishChunk(reason, stop string) string {
	if stop == "" {
		stop = "null"
	}
	return `{"id":"chatcmpl-s1","choices":[{"index":0,"delta":{},"finish_reason":"` + reason + `","stop_reason":` + stop + `}]}`
}

const usageChunk = `{"id":"chatcmpl-s1","choices":[],"usage":{"prompt_tokens":50,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":32}}}`

// convert runs the converter over the stream fed in pieces of every size from
// 1 byte up, so no assertion depends on where the upstream's chunks happened
// to break, and returns the assembled message of the last run.
func convert(t *testing.T, tr *translated, stream string) (*anthropicsse.Message, string) {
	t.Helper()
	var msg *anthropicsse.Message
	var raw string
	for _, size := range []int{1, 2, 7, 64, len(stream) + 1} {
		var out bytes.Buffer
		c := newStreamConverter(&out, func() {}, tr, clock.NewFake(time.Unix(0, 0)), time.Hour,
			func(string, ...any) {})
		c.start("", 10)
		for s := stream; len(s) > 0; {
			k := min(size, len(s))
			_, _ = c.Write([]byte(s[:k]))
			s = s[k:]
		}
		c.Close(false)
		evs, err := anthropicsse.Parse(&out)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		m, err := anthropicsse.Assemble(evs)
		if err != nil {
			t.Fatalf("piece size %d: invalid event stream: %v\n%s", size, err, out.String())
		}
		if msg != nil {
			a, _ := json.Marshal(msg.Content)
			b, _ := json.Marshal(m.Content)
			if !bytes.Equal(a, b) {
				t.Fatalf("piece size %d changed the result:\n%s\n%s", size, a, b)
			}
		}
		msg, raw = m, out.String()
	}
	return msg, raw
}

func TestStreamTextAndThinking(t *testing.T) {
	tr := &translated{clientModel: "claude-x", surfaceThinking: true}
	m, _ := convert(t, tr, upstream(true,
		delta(`{"role":"assistant"}`),
		delta(`{"reasoning_content":"let me "}`),
		delta(`{"reasoning_content":"think"}`),
		delta(`{"content":"Hello"}`),
		delta(`{"content":", world"}`),
		finishChunk("stop", ""), usageChunk))
	if len(m.Content) != 2 || m.Content[0].Type != "thinking" || m.Content[0].Thinking != "let me think" ||
		m.Content[1].Text != "Hello, world" {
		t.Fatalf("content = %+v", m.Content)
	}
	if m.StopReason != "end_turn" || m.Model != "claude-x" {
		t.Errorf("stop %q model %q", m.StopReason, m.Model)
	}
	if m.Usage["input_tokens"] != 18 || m.Usage["cache_read_input_tokens"] != 32 || m.Usage["output_tokens"] != 7 {
		t.Errorf("usage = %v", m.Usage)
	}
}

func TestStreamHidesReasoningUnlessAsked(t *testing.T) {
	m, _ := convert(t, &translated{clientModel: "c"}, upstream(true,
		delta(`{"reasoning":"secret"}`), delta(`{"content":"answer"}`), finishChunk("stop", "")))
	if len(m.Content) != 1 || m.Content[0].Text != "answer" {
		t.Errorf("content = %+v", m.Content)
	}
}

func toolHead(i int, id, name string) string {
	return delta(`{"tool_calls":[{"index":` + string(rune('0'+i)) + `,"id":"` + id + `","type":"function","function":{"name":"` + name + `","arguments":""}}]}`)
}

func toolArgs(i int, frag string) string {
	b, _ := json.Marshal(frag)
	return delta(`{"tool_calls":[{"index":` + string(rune('0'+i)) + `,"function":{"arguments":` + string(b) + `}}]}`)
}

func TestStreamParallelToolCallsSequential(t *testing.T) {
	m, _ := convert(t, &translated{clientModel: "c"}, upstream(true,
		delta(`{"content":"Checking both."}`),
		toolHead(0, "call_a", "get_weather"),
		toolArgs(0, `{"ci`), toolArgs(0, `ty":"Pa`), toolArgs(0, `ris"}`),
		toolHead(1, "call_b", "get_weather"),
		toolArgs(1, `{"city":`), toolArgs(1, `"Rome"}`),
		finishChunk("tool_calls", ""), usageChunk))
	assertTools(t, m, `{"city":"Paris"}`, `{"city":"Rome"}`)
	if m.Content[0].Text != "Checking both." {
		t.Errorf("text block = %+v", m.Content[0])
	}
}

// Some servers interleave parallel calls' argument fragments. Anthropic cannot
// interleave blocks, so the converter must buffer, and every call must still
// arrive whole and in order.
func TestStreamParallelToolCallsInterleaved(t *testing.T) {
	m, _ := convert(t, &translated{clientModel: "c"}, upstream(true,
		toolHead(0, "call_a", "Read"),
		toolHead(1, "call_b", "Grep"),
		toolArgs(0, `{"file_path":`), toolArgs(1, `{"pattern":`),
		toolArgs(0, `"/a"}`), toolArgs(1, `"foo"}`),
		finishChunk("tool_calls", ""), usageChunk))
	if len(m.Content) != 2 {
		t.Fatalf("content = %+v", m.Content)
	}
	if string(m.Content[0].Input) != `{"file_path":"/a"}` || string(m.Content[1].Input) != `{"pattern":"foo"}` ||
		m.Content[0].Name != "Read" || m.Content[1].Name != "Grep" {
		t.Errorf("content = %+v", m.Content)
	}
}

func assertTools(t *testing.T, m *anthropicsse.Message, inputs ...string) {
	t.Helper()
	var got []string
	ids := map[string]bool{}
	for _, b := range m.Content {
		if b.Type == "tool_use" {
			got = append(got, string(b.Input))
			ids[b.ID] = true
		}
	}
	if strings.Join(got, "|") != strings.Join(inputs, "|") {
		t.Errorf("tool inputs %v, want %v", got, inputs)
	}
	if len(ids) != len(inputs) {
		t.Errorf("tool ids not distinct: %v", ids)
	}
	if m.StopReason != "tool_use" {
		t.Errorf("stop_reason %q, want tool_use", m.StopReason)
	}
}

func TestStreamToolCallWithoutArguments(t *testing.T) {
	m, _ := convert(t, &translated{clientModel: "c"}, upstream(true,
		toolHead(0, "call_a", "LS"), finishChunk("tool_calls", "")))
	assertTools(t, m, `{}`)
}

func TestStreamStopSequenceAndLength(t *testing.T) {
	m, _ := convert(t, &translated{clientModel: "c", stopSequences: []string{"###"}}, upstream(true,
		delta(`{"content":"a"}`), finishChunk("stop", `"###"`)))
	if m.StopReason != "stop_sequence" || m.StopSequence != "###" {
		t.Errorf("stop %q seq %q", m.StopReason, m.StopSequence)
	}
	m, _ = convert(t, &translated{clientModel: "c"}, upstream(true,
		delta(`{"content":"a"}`), finishChunk("length", "")))
	if m.StopReason != "max_tokens" {
		t.Errorf("stop %q", m.StopReason)
	}
}

func TestStreamEmptyGenerationStillHasAContentBlock(t *testing.T) {
	m, _ := convert(t, &translated{clientModel: "c"}, upstream(true,
		delta(`{"role":"assistant"}`), finishChunk("stop", "")))
	if len(m.Content) != 1 || m.Content[0].Type != "text" || m.Content[0].Text != "" {
		t.Errorf("content = %+v", m.Content)
	}
}

// A backend that finished but never sent [DONE] still produced a whole reply.
func TestStreamFinishWithoutDoneCompletes(t *testing.T) {
	m, _ := convert(t, &translated{clientModel: "c"}, upstream(false,
		delta(`{"content":"ok"}`), finishChunk("stop", "")))
	if m.Error != nil || m.StopReason != "end_turn" {
		t.Errorf("message = %+v", m)
	}
}

// One that stopped mid-reply did not, and the client must be told.
func TestStreamCutOffEndsWithErrorEvent(t *testing.T) {
	m, _ := convert(t, &translated{clientModel: "c"}, upstream(false,
		delta(`{"content":"partial"}`), toolHead(0, "call_a", "Read"), toolArgs(0, `{"file`)))
	if m.Error == nil || m.Error.Type != "api_error" {
		t.Fatalf("no error event for a truncated stream: %+v", m)
	}
	m, _ = convert(t, &translated{clientModel: "c"}, upstream(true, delta(`{"content":"partial"}`)))
	if m.Error == nil {
		t.Fatalf("[DONE] without a finish reason accepted as complete")
	}
}

func TestStreamUpstreamErrorChunk(t *testing.T) {
	m, _ := convert(t, &translated{clientModel: "c"}, upstream(true,
		delta(`{"content":"a"}`), `{"error":{"message":"engine died","type":"InternalServerError","code":500}}`))
	if m.Error == nil || !strings.Contains(m.Error.Message, "engine died") {
		t.Fatalf("error = %+v", m.Error)
	}
}

// While the backend reasons and the client is shown none of it, pings must
// keep the connection visibly alive.
func TestStreamPingsWhileReasoningIsHidden(t *testing.T) {
	var out bytes.Buffer
	clk := clock.NewFake(time.Unix(0, 0))
	c := newStreamConverter(&out, func() {}, &translated{clientModel: "c"}, clk, 10*time.Second,
		func(string, ...any) {})
	c.start("", 1)
	for range 5 {
		clk.Advance(6 * time.Second)
		_, _ = c.Write([]byte("data: " + delta(`{"reasoning":"..."}`) + "\n\n"))
	}
	_, _ = c.Write([]byte(upstream(true, delta(`{"content":"x"}`), finishChunk("stop", ""))))
	c.Close(false)
	evs, _ := anthropicsse.Parse(&out)
	m, err := anthropicsse.Assemble(evs)
	if err != nil {
		t.Fatal(err)
	}
	// One at start, then one per 10s of silence across 30s.
	if m.Pings < 3 {
		t.Errorf("pings = %d, want >= 3", m.Pings)
	}
}

// A client that hung up gets nothing more written.
func TestStreamCloseAfterClientGoneWritesNothing(t *testing.T) {
	var out bytes.Buffer
	c := newStreamConverter(&out, func() {}, &translated{clientModel: "c"}, clock.Real{}, time.Hour,
		func(string, ...any) {})
	c.start("", 1)
	_, _ = c.Write([]byte("data: " + delta(`{"content":"x"}`) + "\n\n"))
	n := out.Len()
	c.Close(true)
	if out.Len() != n {
		t.Errorf("wrote %q after the client left", out.String()[n:])
	}
}

func TestReplayAsStreamMatchesTheWholeMessage(t *testing.T) {
	tr := &translated{clientModel: "c", surfaceThinking: true}
	whole, _, err := fromOpenAI([]byte(`{"id":"chatcmpl-z","choices":[{"index":0,"message":{"role":"assistant",
	  "content":"hi","reasoning_content":"r","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},
	  "finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}`), tr)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	replayAsStream(&out, whole)
	evs, _ := anthropicsse.Parse(&out)
	m, err := anthropicsse.Assemble(evs)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Content) != 3 || m.Content[0].Thinking != "r" || m.Content[1].Text != "hi" ||
		string(m.Content[2].Input) != `{"a":1}` || m.StopReason != "tool_use" || m.ID != "msg_z" {
		t.Errorf("replayed = %+v", m)
	}
}
