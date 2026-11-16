package mockvllm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func scriptServer(t *testing.T, agentic bool) *httptest.Server {
	t.Helper()
	cfg := DefaultConfig()
	cfg.ModelID = "m"
	cfg.OutputTPS = 1e6
	cfg.Agentic = agentic
	ts := httptest.NewServer(NewServer(NewEngine(cfg)).Handler())
	t.Cleanup(ts.Close)
	return ts
}

func postScript(t *testing.T, ts *httptest.Server, script, body string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(body))
	if script != "" {
		// A header value cannot hold a newline; compact what a test wrote
		// across lines. A malformed script is sent as is.
		var c bytes.Buffer
		if json.Compact(&c, []byte(script)) == nil {
			script = c.String()
		}
		req.Header.Set(ScriptHeader, script)
	}
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

type sseChunk struct {
	Choices []struct {
		Delta struct {
			Content          *string `json:"content"`
			ReasoningContent *string `json:"reasoning_content"`
			Reasoning        *string `json:"reasoning"`
			ToolCalls        []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
		StopReason   *string `json:"stop_reason"`
	} `json:"choices"`
	Usage *usage `json:"usage"`
}

func readChunks(t *testing.T, r io.Reader) ([]sseChunk, bool) {
	t.Helper()
	var out []sseChunk
	done := false
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := strings.TrimPrefix(sc.Text(), "data: ")
		if line == "" {
			continue
		}
		if line == "[DONE]" {
			done = true
			continue
		}
		var c sseChunk
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Fatalf("chunk %q: %v", line, err)
		}
		out = append(out, c)
	}
	return out, done
}

const scriptChatBody = `{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`

// A scripted stream splits every string across chunks, the tool-call head
// carries id and name, and argument fragments reassemble per call — the
// shapes a Chat Completions client has to cope with.
func TestScriptStreamsSplitToolCalls(t *testing.T) {
	ts := scriptServer(t, false)
	resp := postScript(t, ts, `{"reasoning":"think","text":"Hi","chunk_chars":2,"interleave_tools":true,
	  "tool_calls":[{"name":"a","arguments":"{\"x\":1}"},{"id":"given","name":"b","arguments":"{\"y\":22}"}]}`, scriptChatBody)
	chunks, done := readChunks(t, resp.Body)
	if !done {
		t.Fatal("no [DONE]")
	}
	var reasoning, text string
	args := map[int]string{}
	ids := map[int]string{}
	var finish string
	var sawUsage bool
	interleaved, last := false, -1
	for _, c := range chunks {
		if c.Usage != nil {
			sawUsage = true
		}
		for _, ch := range c.Choices {
			d := ch.Delta
			if d.ReasoningContent != nil {
				reasoning += *d.ReasoningContent
			}
			if d.Content != nil {
				text += *d.Content
			}
			for _, tc := range d.ToolCalls {
				if tc.ID != "" {
					ids[tc.Index] = tc.ID
				}
				if tc.Function.Arguments != "" {
					if last >= 0 && last != tc.Index {
						interleaved = true
					}
					last = tc.Index
				}
				args[tc.Index] += tc.Function.Arguments
			}
			if ch.FinishReason != nil {
				finish = *ch.FinishReason
			}
		}
	}
	if reasoning != "think" || text != "Hi" || args[0] != `{"x":1}` || args[1] != `{"y":22}` {
		t.Errorf("reasoning %q text %q args %v", reasoning, text, args)
	}
	if ids[0] == "" || ids[1] != "given" || !interleaved || finish != "tool_calls" || !sawUsage {
		t.Errorf("ids %v interleaved %v finish %q usage %v", ids, interleaved, finish, sawUsage)
	}
}

func TestScriptReasoningFieldStopReasonAndCut(t *testing.T) {
	ts := scriptServer(t, false)
	chunks, _ := readChunks(t, postScript(t, ts,
		`{"reasoning":"r","reasoning_field":"reasoning","text":"t","stop_reason":"END"}`, scriptChatBody).Body)
	var sawReasoning bool
	var stop string
	for _, c := range chunks {
		for _, ch := range c.Choices {
			if ch.Delta.Reasoning != nil && ch.Delta.ReasoningContent == nil {
				sawReasoning = true
			}
			if ch.StopReason != nil {
				stop = *ch.StopReason
			}
		}
	}
	if !sawReasoning || stop != "END" {
		t.Errorf("reasoning field %v stop_reason %q", sawReasoning, stop)
	}

	chunks, done := readChunks(t, postScript(t, ts, `{"text":"abcdefghijkl","cut_after_chunks":3}`, scriptChatBody).Body)
	for _, c := range chunks {
		for _, ch := range c.Choices {
			if ch.FinishReason != nil {
				t.Errorf("cut stream carried a finish reason")
			}
		}
	}
	if done || len(chunks) != 3 {
		t.Errorf("cut stream: %d chunks, done %v", len(chunks), done)
	}
}

func TestScriptErrorAndNonStreaming(t *testing.T) {
	ts := scriptServer(t, false)
	resp := postScript(t, ts, `{"error_status":400,"error_message":"too long"}`, scriptChatBody)
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 400 || !strings.Contains(string(b), "too long") {
		t.Errorf("%d %s", resp.StatusCode, b)
	}
	resp = postScript(t, ts, `{"text":"x","tool_calls":[{"name":"f","arguments":"{}"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	var out struct {
		Choices []struct {
			Message struct {
				Content   *string           `json:"content"`
				ToolCalls []json.RawMessage `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *usage `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	m := out.Choices[0]
	if *m.Message.Content != "x" || len(m.Message.ToolCalls) != 1 || m.FinishReason != "tool_calls" || out.Usage == nil {
		t.Errorf("%+v", out)
	}
	resp = postScript(t, ts, `{not json`, scriptChatBody)
	if resp.StatusCode != 400 {
		t.Errorf("malformed script: %d", resp.StatusCode)
	}
}

// Agentic mode: tools offered -> call the first one with schema-typed
// arguments; a tool result last -> answer in text; no tools -> the ordinary
// synthetic reply, so benchmark traffic is unchanged.
func TestAgenticReplies(t *testing.T) {
	ts := scriptServer(t, true)
	tools := `"tools":[{"type":"function","function":{"name":"Read","parameters":{"type":"object",
	  "properties":{"file_path":{"type":"string"},"limit":{"type":"integer"}},"required":["file_path","limit"]}}}]`
	resp := postScript(t, ts, "", `{"model":"m","chat_template_kwargs":{"thinking":true},`+tools+`,
	  "messages":[{"role":"user","content":"read it"}]}`)
	var out struct {
		Choices []struct {
			Message struct {
				ReasoningContent string `json:"reasoning_content"`
				ToolCalls        []struct {
					Function struct{ Name, Arguments string } `json:"function"`
				} `json:"tool_calls"`
				Content *string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&out)
	m := out.Choices[0].Message
	if len(m.ToolCalls) != 1 || m.ToolCalls[0].Function.Name != "Read" ||
		m.ToolCalls[0].Function.Arguments != `{"file_path":"mock","limit":1}` || m.ReasoningContent == "" {
		t.Errorf("first turn: %+v", out)
	}

	resp = postScript(t, ts, "", `{"model":"m",`+tools+`,"messages":[{"role":"user","content":"read it"},
	  {"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"Read","arguments":"{}"}}]},
	  {"role":"tool","tool_call_id":"c1","content":"hello"}]}`)
	out.Choices = nil
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if c := out.Choices[0]; c.Message.Content == nil || !strings.HasPrefix(*c.Message.Content, "Done") ||
		len(c.Message.ToolCalls) != 0 || c.FinishReason != "stop" {
		t.Errorf("after tool result: %+v", out)
	}

	resp = postScript(t, ts, "", `{"model":"m","max_tokens":3,"messages":[{"role":"user","content":"x"}]}`)
	out.Choices = nil
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if c := out.Choices[0]; c.Message.Content == nil || *c.Message.Content != "tok0 tok1 tok2" {
		t.Errorf("no tools must get the synthetic reply: %+v", out)
	}
}
