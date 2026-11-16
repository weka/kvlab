package anthropicapi

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestFinishReasonMapping(t *testing.T) {
	stops := []string{"END"}
	for _, c := range []struct {
		finish, vllmStop string
		tools            bool
		want, seq        string
	}{
		{"stop", "", false, "end_turn", ""},
		{"stop", `null`, false, "end_turn", ""},
		{"stop", `"END"`, false, "stop_sequence", "END"},
		{"stop", `"OTHER"`, false, "end_turn", ""},
		{"stop", `151643`, false, "end_turn", ""}, // a stop token id, not a sequence
		{"length", "", false, "max_tokens", ""},
		{"tool_calls", "", true, "tool_use", ""},
		{"stop", "", true, "tool_use", ""},
		{"content_filter", "", false, "refusal", ""},
	} {
		got, seq := stopReason(c.finish, json.RawMessage(c.vllmStop), stops, c.tools)
		gotSeq := ""
		if seq != nil {
			gotSeq = *seq
		}
		if got != c.want || gotSeq != c.seq {
			t.Errorf("%s/%s/%v -> %s %q, want %s %q", c.finish, c.vllmStop, c.tools, got, gotSeq, c.want, c.seq)
		}
	}
}

func TestNonStreamingResponse(t *testing.T) {
	tr := &translated{clientModel: "claude-sonnet", surfaceThinking: true}
	body := `{"id":"chatcmpl-abc","model":"backend","choices":[{"index":0,"message":{"role":"assistant",
	  "content":"Let me look.","reasoning_content":"hmm",
	  "tool_calls":[{"id":"call_1","type":"function","function":{"name":"Read","arguments":"{\"file_path\":\"/x\"}"}},
	                {"id":"call_2","type":"function","function":{"name":"Glob","arguments":""}}]},
	  "finish_reason":"tool_calls"}],
	  "usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":64}}}`
	m, warns, err := fromOpenAI([]byte(body), tr)
	if err != nil || len(warns) != 0 {
		t.Fatalf("err %v warnings %v", err, warns)
	}
	b, _ := json.Marshal(m)
	jsonEq(t, json.RawMessage(b), `{"id":"msg_abc","type":"message","role":"assistant","model":"claude-sonnet",
	  "content":[{"type":"thinking","thinking":"hmm","signature":""},
	             {"type":"text","text":"Let me look."},
	             {"type":"tool_use","id":"call_1","name":"Read","input":{"file_path":"/x"}},
	             {"type":"tool_use","id":"call_2","name":"Glob","input":{}}],
	  "stop_reason":"tool_use","stop_sequence":null,
	  "usage":{"input_tokens":36,"cache_creation_input_tokens":0,"cache_read_input_tokens":64,"output_tokens":20}}`)

	tr.surfaceThinking = false
	m, _, _ = fromOpenAI([]byte(body), tr)
	if m.Content[0].Type != "text" {
		t.Errorf("reasoning surfaced although not requested: %+v", m.Content[0])
	}
}

func TestNonStreamingReasoningFieldAndBadArguments(t *testing.T) {
	tr := &translated{clientModel: "c", surfaceThinking: true}
	body := `{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning":"r",
	  "tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{not json"}}]},"finish_reason":"tool_calls"}]}`
	m, warns, err := fromOpenAI([]byte(body), tr)
	if err != nil {
		t.Fatal(err)
	}
	if len(warns) != 1 {
		t.Errorf("malformed arguments not warned about: %v", warns)
	}
	if *m.Content[0].Thinking != "r" || string(m.Content[1].Input) != "{}" {
		t.Errorf("content = %+v", m.Content)
	}
	if _, _, err := fromOpenAI([]byte(`{"choices":[]}`), tr); err == nil {
		t.Error("empty choices accepted")
	}
	if _, _, err := fromOpenAI([]byte(`<html>`), tr); err == nil {
		t.Error("non-JSON accepted")
	}
}

func TestEmptyCompletionHasEmptyContentArray(t *testing.T) {
	m, _, err := fromOpenAI([]byte(`{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"length"}]}`),
		&translated{clientModel: "c"})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(m)
	if !strings.Contains(string(b), `"content":[]`) || *m.StopReason != "max_tokens" {
		t.Errorf("got %s", b)
	}
}

func TestErrorEnvelopeAndMessages(t *testing.T) {
	for status, typ := range map[int]string{400: "invalid_request_error", 401: "authentication_error",
		403: "permission_error", 404: "not_found_error", 413: "request_too_large",
		429: "rate_limit_error", 500: "api_error", 502: "api_error", 503: "overloaded_error", 529: "overloaded_error"} {
		rec := httptest.NewRecorder()
		writeError(rec, status, "boom")
		var env struct {
			Type  string `json:"type"`
			Error struct{ Type, Message string }
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
			t.Fatal(err)
		}
		if rec.Code != status || env.Type != "error" || env.Error.Type != typ || env.Error.Message != "boom" {
			t.Errorf("%d: %d %s", status, rec.Code, rec.Body)
		}
		if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
			t.Errorf("content type %q", ct)
		}
	}
	for body, want := range map[string]string{
		`{"error":{"message":"bad thing","type":"BadRequestError","code":400}}`: "bad thing",
		`{"message":"plain"}`:       "plain",
		`{"error":"just a string"}`: "just a string",
		`upstream exploded`:         "upstream exploded",
		`{"detail":"nope"}`:         `"nope"`,
	} {
		if got := upstreamErrorMessage([]byte(body)); got != want {
			t.Errorf("%s -> %q, want %q", body, got, want)
		}
	}
	got := translateErrorMessage(http.StatusBadRequest,
		"This model's maximum context length is 1000 tokens. However, you requested 2000 tokens.")
	if !strings.HasPrefix(got, "prompt is too long: ") {
		t.Errorf("context-length error not reworded: %q", got)
	}
	if got := translateErrorMessage(http.StatusBadRequest, "other"); got != "other" {
		t.Errorf("unrelated error reworded: %q", got)
	}
}
