package benchmark

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// overflowSession is one instance of four turns; turn 2's prompt is 300k
// tokens in the capture.
func overflowSession() RouterReplaySession {
	var reqs []RouterReplayRequest
	for i := 1; i <= 4; i++ {
		in := 1000 * i
		if i >= 2 {
			in = 300000
		}
		reqs = append(reqs, RouterReplayRequest{RequestID: uint64(i), OutputTokens: 10, InputTokens: in,
			Messages: []RouterReplayMessage{userText(fmt.Sprintf("turn-%d", i), 40)}})
	}
	return RouterReplaySession{SessionID: "s-long", Instances: []RouterReplayInstance{{InstanceID: "i1", Role: "main", Requests: reqs}}}
}

// overflowServer answers the first request and rejects every later one with
// vLLM's context-length 400.
func overflowServer(t *testing.T) (*httptest.Server, *atomic.Int64) {
	var n atomic.Int64
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if n.Add(1) > 1 {
			http.Error(w, `{"error":{"message":"This model's maximum context length is 262144 tokens. However, you requested 300010 tokens."}}`, 400)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":30,"completion_tokens":5,"total_tokens":35}}`)
	}))
	t.Cleanup(ts.Close)
	return ts, &n
}

func readOverflowRows(t *testing.T, dir string) []overflowRecord {
	t.Helper()
	files, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	var out []overflowRecord
	for _, f := range files {
		c, err := readJSONLContents(f)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, c.overflows...)
	}
	return out
}

func runOverflowSession(t *testing.T, limitContext int) (*autoState, string, *atomic.Int64) {
	t.Helper()
	ts, n := overflowServer(t)
	cfg := AutoBenchmarkConfig{
		Model:         fmt.Sprintf("dynamic/%s,type=openai,model=test-model", ts.URL),
		ReplayNoStamp: true,
		LimitContext:  limitContext,
	}
	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "test-model", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	st := newTestAutoState(4)
	runRouterReplaySession(context.Background(), cfg, st, rdw, overflowSession(), 1,
		endpointPicker{}, 30*time.Second, strings.Repeat("session-docs ", 100), newConcurrencyGate(4, false))
	rdw.close()
	return st, dir, n
}

func TestOverflowServer400IsCountedAndRecorded(t *testing.T) {
	st, dir, n := runOverflowSession(t, 0)
	if got := n.Load(); got != 2 {
		t.Errorf("server saw %d requests, want 2: the 400 retires the instance", got)
	}
	s := st.overflow.summary()
	if s.Sessions != 1 || s.Server != 1 || s.LimitContext != 0 || s.Requests != 3 || s.MaxContext != 262144 {
		t.Fatalf("overflow %+v, want 1 server retirement dropping 3 requests at 262144", s)
	}
	if st.totalErrors.Load() != 0 {
		t.Errorf("overflow counted as %d errors", st.totalErrors.Load())
	}
	rows := readOverflowRows(t, dir)
	if len(rows) != 1 {
		t.Fatalf("overflow rows %d, want 1", len(rows))
	}
	r := rows[0]
	if r.ErrorKind != "context_overflow" || r.Reason != "server_400" || r.Turn != 2 || r.PromptTokens != 300000 ||
		r.MaxContext != 262144 || r.RequestsDropped != 3 || !strings.Contains(r.ErrorMessage, "maximum context length") {
		t.Errorf("row %+v", r)
	}
	// Request rows are only the completed request: the overflow row is not one.
	c, _ := readJSONLContents(mustGlobOne(t, dir))
	if len(c.records) != 1 {
		t.Errorf("request rows %d, want 1", len(c.records))
	}
}

func TestOverflowLimitContextIsCountedAndRecorded(t *testing.T) {
	st, dir, n := runOverflowSession(t, 200000)
	if got := n.Load(); got != 1 {
		t.Errorf("server saw %d requests, want 1: --limit-context skips before sending", got)
	}
	s := st.overflow.summary()
	if s.Sessions != 1 || s.LimitContext != 1 || s.Server != 0 || s.Requests != 3 || s.MaxContext != 200000 {
		t.Fatalf("overflow %+v, want 1 --limit-context retirement dropping 3 at 200000", s)
	}
	rows := readOverflowRows(t, dir)
	if len(rows) != 1 || rows[0].Reason != "limit_context" || rows[0].ErrorKind != "context_overflow" {
		t.Errorf("rows %+v", rows)
	}
}

func mustGlobOne(t *testing.T, dir string) string {
	t.Helper()
	files, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if len(files) != 1 {
		t.Fatalf("jsonl files %v", files)
	}
	return files[0]
}

func TestOverflowSummaryAndProgressLine(t *testing.T) {
	var c overflowCounts
	if c.summary().String() != "" || overflowTok(0) != "" {
		t.Error("nothing retired still reported")
	}
	c.add(overflowRecord{Reason: overflowReasonServer, RequestsDropped: 5, MaxContext: 262144})
	c.add(overflowRecord{Reason: overflowReasonLimit, RequestsDropped: 2})
	s := c.summary()
	if got := s.String(); got != "2 session(s) retired (server 400: 1, --limit-context: 1), 7 request(s) never sent, max context 262144" {
		t.Errorf("summary %q", got)
	}
	if !strings.Contains(s.Warning, "round server limit") {
		t.Errorf("warning %q lacks the round-limit note", s.Warning)
	}
	line := renderModelOneLiner(&displaySnapshot{totalErrors: 0, overflow: 2})
	if !strings.Contains(line, "errors=0 overflow=2") {
		t.Errorf("progress line %q lacks overflow=2", line)
	}
}

func TestOverflowInVisualizeAndMerge(t *testing.T) {
	_, dir, _ := runOverflowSession(t, 0)
	html, err := GenerateVisualization(dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(html)
	page := string(b)
	for _, want := range []string{`"overflow":{"sessions":1,"requests":3,"server400":1`, "round server limit", `id="overflowWarn"`, "context_overflow_sessions"} {
		if !strings.Contains(page, want) {
			t.Errorf("report lacks %q", want)
		}
	}
	out := t.TempDir()
	mhtml, err := GenerateVisualizationMerged([]string{dir}, []string{"arm"}, out, 0, 0, "")
	if err != nil {
		t.Fatal(err)
	}
	mb, _ := os.ReadFile(mhtml)
	if !strings.Contains(string(mb), `"overflow":{"sessions":1`) {
		t.Error("merged report lost the overflow count")
	}
	if rows := readOverflowRows(t, out); len(rows) != 1 {
		t.Errorf("merged jsonl overflow rows %d, want 1", len(rows))
	}
}

func TestContextFitStartupWarning(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"data":[{"id":"m","max_model_len":262144}]}`)
	}))
	defer ts.Close()
	limit, err := fetchMaxModelLen(ts.URL)
	if err != nil || limit != 262144 {
		t.Fatalf("max_model_len %d, %v", limit, err)
	}
	// Ten sessions of two requests; the second request of six sessions is
	// over the limit.
	path := filepath.Join(t.TempDir(), "replay.jsonl")
	f, _ := os.Create(path)
	w := bufio.NewWriter(f)
	fmt.Fprintln(w, `{"_schema":"replay-v5","summary":{"sessions":10}}`)
	for i := range 10 {
		second := 100000
		if i < 6 {
			second = 400000
		}
		sess := map[string]any{"session_id": i, "instances": []any{map[string]any{"requests": []any{
			map[string]any{"input_tokens": 1000}, map[string]any{"input_tokens": 1000, "cache_read_tokens": second},
		}}}}
		b, _ := json.Marshal(sess)
		fmt.Fprintln(w, string(b))
	}
	w.Flush()
	f.Close()
	fit, err := scanCorpusFit(path, limit)
	if err != nil {
		t.Fatal(err)
	}
	if fit.Sessions != 10 || fit.Requests != 20 || fit.RequestsOver != 6 || fit.SessionsOver != 6 || fit.RequestsDropped != 6 {
		t.Fatalf("fit %+v", fit)
	}
	msg := fit.warning(limit)
	for _, want := range []string{"WARNING", "max_model_len 262144", "6 of 20 corpus requests", "6 of 10 sessions (60%)"} {
		if !strings.Contains(msg, want) {
			t.Errorf("warning %q lacks %q", msg, want)
		}
	}
	if (corpusFit{Requests: 3}).warning(limit) != "" {
		t.Error("warned with nothing over the limit")
	}
}
