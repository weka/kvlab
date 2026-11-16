package serve_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/mockvllm"
	"github.com/weka/kvlab/router/internal/testutil/anthropicsse"
	"github.com/weka/kvlab/router/serve"
)

// agentFleet is a mock fleet in agentic mode: tool-offering requests get
// tool-using replies, so a client can run whole loops through the router.
func agentFleet(t *testing.T, n int) ([]string, []*mockvllm.Server) {
	t.Helper()
	var urls []string
	var servers []*mockvllm.Server
	for range n {
		cfg := mockvllm.DefaultConfig()
		cfg.ModelID = "backend-model"
		cfg.MaxConcurrency = 64
		cfg.OutputTPS = 100_000
		cfg.Agentic = true
		srv := mockvllm.NewServer(mockvllm.NewEngine(cfg))
		ts := httptest.NewServer(srv.Handler())
		t.Cleanup(ts.Close)
		urls = append(urls, ts.URL)
		servers = append(servers, srv)
	}
	return urls, servers
}

func anthropicRouter(t *testing.T, urls []string, mod func(*serve.Options)) *httptest.Server {
	t.Helper()
	opts := serve.Options{
		AutoModel: "off",
		Routes:    []serve.Route{{Name: "fleet", Patterns: "*", Endpoints: urls}},
		Anthropic: &serve.AnthropicOptions{
			ModelMap:       []string{"claude=backend-model"},
			ThinkingKwargs: []string{"thinking"},
		},
	}
	if mod != nil {
		mod(&opts)
	}
	rt := startRouter(t, opts)
	// Ready once every backend is healthy.
	eventually(t, func() bool {
		resp, err := http.Get(rt.URL + "/workers")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode == http.StatusOK && strings.Count(string(b), `"health":"healthy"`) >= len(urls)
	})
	return rt
}

// messagesCall posts an Anthropic request the way Claude Code does: x-api-key,
// anthropic-version and anthropic-beta headers, optionally a mock script.
func messagesCall(t *testing.T, ctx context.Context, rt *httptest.Server, body, script string) *http.Response {
	t.Helper()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, rt.URL+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Api-Key", "sk-ant-client-key")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	req.Header.Set("Anthropic-Beta", "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14")
	if script != "" {
		var c bytes.Buffer
		if err := json.Compact(&c, []byte(script)); err != nil {
			t.Fatal(err)
		}
		req.Header.Set(mockvllm.ScriptHeader, c.String())
	}
	resp, err := rt.Client().Do(req)
	if err != nil {
		t.Fatalf("POST /v1/messages: %v", err)
	}
	return resp
}

func streamMessage(t *testing.T, resp *http.Response) *anthropicsse.Message {
	t.Helper()
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d: %s", resp.StatusCode, b)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content type %q", ct)
	}
	evs, err := anthropicsse.Parse(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	m, err := anthropicsse.Assemble(evs)
	if err != nil {
		t.Fatalf("invalid Anthropic event stream: %v", err)
	}
	return m
}

const weatherTools = `"tools":[{"name":"get_weather","description":"Weather for a city",
  "input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},
  "cache_control":{"type":"ephemeral"}}]`

func TestAnthropicNonStreamingThroughRouter(t *testing.T) {
	urls, _ := agentFleet(t, 2)
	rt := anthropicRouter(t, urls, nil)
	resp := messagesCall(t, context.Background(), rt, `{"model":"claude-sonnet-4-5","max_tokens":100,
	  "system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}],
	  "messages":[{"role":"user","content":"Hello"}]}`,
		`{"text":"Hi there.","reasoning":"hidden"}`)
	defer resp.Body.Close()
	var m map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&m)
	if resp.StatusCode != 200 || m["type"] != "message" || m["model"] != "claude-sonnet-4-5" ||
		m["stop_reason"] != "end_turn" || !strings.HasPrefix(m["id"].(string), "msg_") {
		t.Fatalf("%d %v", resp.StatusCode, m)
	}
	content := m["content"].([]any)
	if len(content) != 1 || content[0].(map[string]any)["text"] != "Hi there." {
		t.Errorf("content = %v (reasoning must stay hidden without thinking)", content)
	}
	u := m["usage"].(map[string]any)
	if u["input_tokens"].(float64) <= 0 || u["output_tokens"].(float64) <= 0 {
		t.Errorf("usage = %v", u)
	}
}

func TestAnthropicStreamingParallelToolCallsThroughRouter(t *testing.T) {
	urls, _ := agentFleet(t, 2)
	rt := anthropicRouter(t, urls, nil)
	resp := messagesCall(t, context.Background(), rt, `{"model":"claude-sonnet-4-5","max_tokens":1000,"stream":true,
	  "thinking":{"type":"enabled","budget_tokens":500},`+weatherTools+`,
	  "messages":[{"role":"user","content":"Weather in Paris and Rome?"}]}`,
		`{"reasoning":"Two cities, two calls.","text":"Checking.","chunk_chars":2,"interleave_tools":true,
		  "tool_calls":[{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"},
		                {"name":"get_weather","arguments":"{\"city\":\"Rome\"}"}]}`)
	m := streamMessage(t, resp)
	if m.Model != "claude-sonnet-4-5" || m.StopReason != "tool_use" {
		t.Fatalf("model %q stop %q", m.Model, m.StopReason)
	}
	var kinds []string
	for _, b := range m.Content {
		kinds = append(kinds, b.Type)
	}
	if strings.Join(kinds, ",") != "thinking,text,tool_use,tool_use" {
		t.Fatalf("blocks %v", kinds)
	}
	if m.Content[0].Thinking != "Two cities, two calls." || m.Content[1].Text != "Checking." ||
		string(m.Content[2].Input) != `{"city":"Paris"}` || string(m.Content[3].Input) != `{"city":"Rome"}` ||
		m.Content[2].ID == m.Content[3].ID {
		t.Errorf("content = %+v", m.Content)
	}
	if m.Usage["output_tokens"] <= 0 || m.Pings < 1 {
		t.Errorf("usage %v pings %d", m.Usage, m.Pings)
	}
}

// A whole agent loop: the client executes each tool_use and sends back a
// tool_result until the model ends its turn — the multi-turn shape Claude
// Code drives, with both the tool_use -> tool_calls and the tool_result ->
// role=tool translations on every round trip.
func TestAnthropicAgentLoopRoundTrips(t *testing.T) {
	urls, _ := agentFleet(t, 2)
	rt := anthropicRouter(t, urls, nil)
	msgs := []any{map[string]any{"role": "user", "content": "What is the weather in Oslo?"}}
	tools := json.RawMessage(`[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]`)
	for _, stream := range []bool{true, false} {
		rounds := 0
		for turn := 0; ; turn++ {
			if turn > 4 {
				t.Fatal("the loop never ended")
			}
			body, _ := json.Marshal(map[string]any{"model": "claude-haiku-4-5", "max_tokens": 200,
				"stream": stream, "tools": tools, "messages": msgs})
			resp := messagesCall(t, context.Background(), rt, string(body), "")
			var content []map[string]any
			var stop string
			if stream {
				m := streamMessage(t, resp)
				stop = m.StopReason
				for _, b := range m.Content {
					raw, _ := json.Marshal(b)
					var mb map[string]any
					_ = json.Unmarshal(raw, &mb)
					content = append(content, mb)
				}
			} else {
				var m struct {
					Content    []map[string]any `json:"content"`
					StopReason string           `json:"stop_reason"`
				}
				_ = json.NewDecoder(resp.Body).Decode(&m)
				resp.Body.Close()
				content, stop = m.Content, m.StopReason
			}
			msgs = append(msgs, map[string]any{"role": "assistant", "content": content})
			if stop != "tool_use" {
				if stop != "end_turn" || content[len(content)-1]["type"] != "text" {
					t.Fatalf("final turn: stop %q content %v", stop, content)
				}
				break
			}
			var results []any
			for _, b := range content {
				if b["type"] == "tool_use" {
					results = append(results, map[string]any{"type": "tool_result",
						"tool_use_id": b["id"], "content": []any{map[string]any{"type": "text", "text": "Sunny, 12C"}}})
				}
			}
			msgs = append(msgs, map[string]any{"role": "user", "content": results})
			rounds++
		}
		if rounds < 1 {
			t.Errorf("stream=%v: no tool round trip happened", stream)
		}
		msgs = msgs[:1]
	}
}

func TestAnthropicErrorsAreAnthropicShaped(t *testing.T) {
	urls, _ := agentFleet(t, 1)
	rt := anthropicRouter(t, urls, func(o *serve.Options) {
		o.Routes = []serve.Route{{Name: "fleet", Patterns: "claude,backend", Endpoints: urls}}
	})
	check := func(name string, resp *http.Response, status int, typ, contains string) {
		t.Helper()
		defer resp.Body.Close()
		var env struct {
			Type  string `json:"type"`
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&env)
		if resp.StatusCode != status || env.Type != "error" || env.Error.Type != typ ||
			!strings.Contains(env.Error.Message, contains) {
			t.Errorf("%s: %d %+v", name, resp.StatusCode, env)
		}
	}
	ctx := context.Background()
	check("backend 400", messagesCall(t, ctx, rt, `{"model":"claude-x","max_tokens":10,"stream":true,
	  "messages":[{"role":"user","content":"x"}]}`,
		`{"error_status":400,"error_message":"This model's maximum context length is 8 tokens."}`),
		400, "invalid_request_error", "prompt is too long")
	check("invalid request", messagesCall(t, ctx, rt, `{"model":"claude-x","max_tokens":10,"messages":[]}`, ""),
		400, "invalid_request_error", "messages")
	check("no route", messagesCall(t, ctx, rt, `{"model":"gpt-4","max_tokens":10,
	  "messages":[{"role":"user","content":"x"}]}`, ""),
		404, "not_found_error", "gpt-4")
	check("backend 500", messagesCall(t, ctx, rt, `{"model":"claude-x","max_tokens":10,
	  "messages":[{"role":"user","content":"x"}]}`, `{"error_status":500,"error_message":"engine fell over"}`),
		500, "api_error", "engine fell over")
}

// A stream the backend abandons mid-reply ends with an error event, never a
// message_stop that would pass a truncated reply off as complete.
func TestAnthropicTruncatedBackendStream(t *testing.T) {
	urls, _ := agentFleet(t, 1)
	rt := anthropicRouter(t, urls, nil)
	resp := messagesCall(t, context.Background(), rt, `{"model":"claude-x","max_tokens":100,"stream":true,
	  "messages":[{"role":"user","content":"x"}]}`, `{"text":"abcdefghijklmnop","cut_after_chunks":4}`)
	m := streamMessage(t, resp)
	if m.Error == nil {
		t.Fatalf("truncated stream ended cleanly: %+v", m)
	}
}

// A client that hangs up mid-stream must free the backend: the request is
// cancelled upstream rather than generated to the end for nobody.
func TestAnthropicClientDisconnectMidStreamCancelsBackend(t *testing.T) {
	urls, servers := agentFleet(t, 1)
	rt := anthropicRouter(t, urls, nil)
	ctx, cancel := context.WithCancel(context.Background())
	resp := messagesCall(t, ctx, rt, `{"model":"claude-x","max_tokens":100,"stream":true,
	  "messages":[{"role":"user","content":"x"}]}`,
		`{"text":"`+strings.Repeat("y", 600)+`","chunk_chars":3,"chunk_delay":20000000}`) // 200 chunks x 20ms
	br := bufio.NewReader(resp.Body)
	deltas := 0
	for deltas < 3 {
		line, err := br.ReadString('\n')
		if err != nil {
			t.Fatalf("stream ended early: %v", err)
		}
		if strings.Contains(line, "text_delta") {
			deltas++
		}
	}
	if got := servers[0].Engine().Stats().Inflight; got != 1 {
		t.Fatalf("backend inflight = %d mid-stream, want 1", got)
	}
	start := time.Now()
	cancel()
	resp.Body.Close()
	eventually(t, func() bool { return servers[0].Engine().Stats().Inflight == 0 })
	if took := time.Since(start); took > 2*time.Second {
		t.Errorf("backend freed only after %v; the stream had ~4s left", took)
	}
	if g := servers[0].Engine().Stats().GeneratedTokens; g != 0 {
		t.Errorf("backend recorded a completed generation (%d tokens) for an abandoned stream", g)
	}
}

func TestAnthropicCountTokens(t *testing.T) {
	urls, _ := agentFleet(t, 1)
	rt := anthropicRouter(t, urls, nil)
	req, _ := http.NewRequest(http.MethodPost, rt.URL+"/v1/messages/count_tokens", strings.NewReader(
		`{"model":"claude-x","system":"`+strings.Repeat("s", 400)+`",`+weatherTools+`,
		  "messages":[{"role":"user","content":"hello"}]}`))
	req.Header.Set("Anthropic-Version", "2023-06-01")
	resp, err := rt.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]int
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if resp.StatusCode != 200 || out["input_tokens"] < 100 {
		t.Errorf("%d %v", resp.StatusCode, out)
	}
}

// Translated traffic is cache routed like native chat completions: the same
// conversation keeps landing on the backend that holds its prefix.
func TestAnthropicTranslatedTrafficIsCacheRouted(t *testing.T) {
	urls, servers := agentFleet(t, 3)
	rt := anthropicRouter(t, urls, nil)
	body := `{"model":"claude-x","max_tokens":4,"system":"` + strings.Repeat("shared system prompt ", 300) + `",
	  "messages":[{"role":"user","content":"hello"}]}`
	for range 6 {
		resp := messagesCall(t, context.Background(), rt, body, "")
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("status %d", resp.StatusCode)
		}
	}
	busy := 0
	for _, s := range servers {
		if s.Engine().Stats().Admitted > 0 {
			busy++
		}
	}
	if busy != 1 {
		t.Errorf("6 identical conversations spread over %d backends, want 1", busy)
	}
}

// The router authenticates the caller with its own key and the backend with
// the pool's credential; the client's Anthropic key never leaves the router,
// and the client's anthropic-beta is not forwarded. (A credentialed route adds
// its own anthropic-version, which an OpenAI backend ignores.)
func TestAnthropicAuthAndBackendCredential(t *testing.T) {
	var mu sync.Mutex
	var seen http.Header
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		seen = r.Header.Clone()
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}`)
	}))
	t.Cleanup(backend.Close)
	dir := t.TempDir()
	cred := filepath.Join(dir, "backend.key")
	if err := os.WriteFile(cred, []byte("backend-secret\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	rt := startRouter(t, serve.Options{
		AutoModel: "off",
		APIKey:    "sk-ant-client-key",
		Routes: []serve.Route{{Name: "hosted", Patterns: "*", Endpoints: []string{backend.URL},
			CredentialFile: cred, Passive: true}},
		Anthropic: &serve.AnthropicOptions{ModelMap: []string{"*=backend-model"}},
	})
	resp := messagesCall(t, context.Background(), rt, `{"model":"claude-x","max_tokens":5,
	  "messages":[{"role":"user","content":"x"}]}`, "")
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(string(b), `"text":"ok"`) {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	mu.Lock()
	h := seen
	mu.Unlock()
	if h.Get("Authorization") != "Bearer backend-secret" {
		t.Errorf("backend saw Authorization %q", h.Get("Authorization"))
	}
	for k, v := range h {
		if strings.Contains(strings.Join(v, " "), "sk-ant-client-key") || strings.EqualFold(k, "Anthropic-Beta") {
			t.Errorf("forwarded %s: %v", k, v)
		}
	}

	req, _ := http.NewRequest(http.MethodPost, rt.URL+"/v1/messages",
		strings.NewReader(`{"model":"claude-x","max_tokens":5,"messages":[{"role":"user","content":"x"}]}`))
	req.Header.Set("X-Api-Key", "wrong")
	resp, err := rt.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	b, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 401 || !strings.Contains(string(b), `"authentication_error"`) {
		t.Errorf("bad key: %d %s", resp.StatusCode, b)
	}
}

// With the mode off, /v1/messages is what it always was: proxied untranslated.
func TestAnthropicModeOffLeavesMessagesUntranslated(t *testing.T) {
	urls, _ := mockFleet(t, 1, mockvllm.SurfaceAnthropic)
	rt := startRouter(t, serve.Options{AutoModel: "off",
		Routes: []serve.Route{{Name: "fleet", Patterns: "*", Endpoints: urls}}})
	eventually(t, func() bool {
		code, _ := post(t, rt, "/v1/messages", `{"model":"local-model","max_tokens":3,"messages":[{"role":"user","content":"x"}]}`)
		return code == 200
	})
	_, m := post(t, rt, "/v1/messages", `{"model":"local-model","max_tokens":3,"messages":[{"role":"user","content":"x"}]}`)
	if !strings.HasPrefix(m["id"].(string), "msg-mock-") {
		t.Errorf("not the backend's own messages answer: %v", m)
	}
}
