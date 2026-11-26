package benchmark

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// The captured input_tokens is the whole prompt, so --limit-context compares
// it alone: a warm turn whose cache read is most of its prompt is judged by
// its real size, not by its cached part counted twice.
func TestExceedsLimitContextUsesTheWholeCapturedPrompt(t *testing.T) {
	warm := RouterReplayRequest{InputTokens: 200_000, PrefillTokens: 10_000, CacheReadTokens: 190_000}
	if exceedsLimitContext(warm, 230_000) {
		t.Error("a 200k warm turn was cut at a 230k limit")
	}
	if !exceedsLimitContext(RouterReplayRequest{InputTokens: 230_001}, 230_000) {
		t.Error("a 230,001-token prompt passed a 230k limit")
	}
	if exceedsLimitContext(RouterReplayRequest{InputTokens: 1 << 30}, 0) {
		t.Error("limit 0 must be off")
	}
}

func limitContextSession(input ...int) RouterReplaySession {
	var reqs []RouterReplayRequest
	var msgs []RouterReplayMessage
	for i, n := range input {
		msgs = append(msgs, userText(fmt.Sprintf("turn-%d", i), 40))
		reqs = append(reqs, RouterReplayRequest{RequestID: uint64(i + 1), Ts: "2026-05-12T08:50:06Z",
			OutputTokens: 10, InputTokens: n, CacheReadTokens: n * 9 / 10, PrefillTokens: n / 10,
			Messages: append([]RouterReplayMessage(nil), msgs...)})
	}
	return RouterReplaySession{SessionID: "s", StartTs: "2026-05-12T08:50:06Z",
		Instances: []RouterReplayInstance{{InstanceID: "i1", Role: "main", Requests: reqs}}}
}

// End to end: a warm turn under the limit is sent; the first turn over it
// retires the instance, which is counted by reason, written to the request
// data, and shown by the progress line and the report.
func TestLimitContextRetiresAndCounts(t *testing.T) {
	var sent atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		sent.Add(1)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`)
	}))
	defer ts.Close()
	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "m", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	cfg := AutoBenchmarkConfig{Model: fmt.Sprintf("dynamic/%s,type=openai,model=test-model", ts.URL),
		ReplayNoStamp: true, LimitContext: 230_000}
	st := newTestAutoState(4)
	// 1k, then a 200k warm turn (190k cache read: 390k under the old sum),
	// then 240k over the limit, then a turn that must never be sent.
	runRouterReplaySession(context.Background(), cfg, st, rdw, limitContextSession(1_000, 200_000, 240_000, 250_000), 1,
		endpointPicker{}, 30*time.Second, strings.Repeat("docs ", 100), newConcurrencyGate(4, false))
	_ = rdw.close()

	if n := sent.Load(); n != 2 {
		t.Fatalf("%d requests sent, want 2 (the 1k and 200k turns)", n)
	}
	if got := st.retired.load(); got != (retireTotals{LimitContext: 1}) {
		t.Fatalf("retired = %+v, want one over --limit-context", got)
	}
	if line := renderModelOneLiner(&displaySnapshot{retired: st.retired.load()}); !strings.Contains(line, " retired=limit:1,overflow:0") {
		t.Errorf("progress line %q lacks retired=limit:1,overflow:0", line)
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	c, err := readJSONLContents(files[0])
	if err != nil || len(c.retired) != 1 || c.retired[0].Reason != retireLimitContext ||
		c.retired[0].Turn != 3 || c.retired[0].CapturedInputTokens != 240_000 || len(c.records) != 2 {
		t.Fatalf("request data: %d rows, retired %+v, %v", len(c.records), c.retired, err)
	}
	html, err := GenerateVisualization(dir, 4)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(html)
	if !strings.Contains(string(b), `"retired":{"limit_context":1}`) {
		t.Error("report does not carry the arm's retirement counts")
	}
}

// A server context-length 400 retires the instance too, counted apart.
func TestContextOverflowRetiresAndCounts(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if strings.Count(string(body), `"role":"user"`) >= 2 {
			http.Error(w, `{"error":{"message":"This model's maximum context length is 1000 tokens."}}`, http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`)
	}))
	defer ts.Close()
	cfg := AutoBenchmarkConfig{Model: fmt.Sprintf("dynamic/%s,type=openai,model=test-model", ts.URL), ReplayNoStamp: true}
	st := newTestAutoState(4)
	runRouterReplaySession(context.Background(), cfg, st, nil, limitContextSession(500, 900, 1_500), 1,
		endpointPicker{}, 30*time.Second, strings.Repeat("docs ", 100), newConcurrencyGate(4, false))
	if got := st.retired.load(); got != (retireTotals{ContextOverflow: 1}) {
		t.Fatalf("retired = %+v, want one server context overflow", got)
	}
	if got := st.retired.load().summary(); got != "1 session instances: 0 over --limit-context, 1 by server context-length 400" {
		t.Errorf("summary = %q", got)
	}
}
