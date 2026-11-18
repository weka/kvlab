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
	"testing"
	"time"

	"github.com/weka/kvlab/llm"
)

// slowServer streams OpenAI chunks: after headers it waits firstAfter, then
// sends n tokens every gap (n < 0: forever), then either finishes or, with
// stall, hangs until the client gives up. status != 200 answers that instead.
type slowServer struct {
	status     int
	noHeaders  bool
	firstAfter time.Duration
	n          int
	gap        time.Duration
	stall      bool
}

func (s slowServer) start(t *testing.T) (func(), *replayPoster) {
	t.Helper()
	quit := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.status != 0 && s.status != 200 {
			http.Error(w, "boom", s.status)
			return
		}
		wait := func(d time.Duration) bool {
			select {
			case <-time.After(d):
				return true
			case <-r.Context().Done():
				return false
			case <-quit:
				return false
			}
		}
		if s.noHeaders {
			wait(time.Hour)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		if !wait(s.firstAfter) {
			return
		}
		for i := 0; s.n < 0 || i < s.n; i++ {
			if i > 0 && !wait(s.gap) {
				return
			}
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"t%d \"}}]}\n\n", i)
			w.(http.Flusher).Flush()
		}
		if s.stall {
			wait(time.Hour)
			return
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	p, err := newReplayPoster(fmt.Sprintf("dynamic/%s,type=openai,model=m", ts.URL), llm.APIKeys{OpenAI: "k"}, "", "", false, 0, 0, 0, nil, nil)
	stop := func() {
		close(quit)
		ts.Close()
	}
	if err != nil {
		stop()
		t.Fatal(err)
	}
	return stop, p
}

func runWatched(t *testing.T, s slowServer, global, ttft, idle time.Duration) RequestMetrics {
	t.Helper()
	stop, p := s.start(t)
	defer stop()
	ctx, cancel := withRequestDeadlines(context.Background(), global, ttft, idle)
	defer cancel()
	return p.do(ctx, minReplayReq(true), strings.Repeat("x", 300), 1, "s1", "i1", 1,
		&autoState{stream: newCompletionStream(200)}, nil)
}

func TestRequestDeadlines(t *testing.T) {
	ms := time.Millisecond
	for _, c := range []struct {
		name               string
		srv                slowServer
		global, ttft, idle time.Duration
		kind, msg          string
	}{
		{"no first token, headers sent", slowServer{firstAfter: time.Hour}, time.Minute, 150 * ms, time.Minute, "ttft", "ttft timeout 0.15s"},
		{"no first token, no headers", slowServer{noHeaders: true}, time.Minute, 150 * ms, time.Minute, "ttft", "ttft timeout 0.15s"},
		{"stall mid-stream", slowServer{n: 3, gap: 10 * ms, stall: true}, time.Minute, time.Minute, 150 * ms, "idle", "stream idle 0.15s after 3 tokens"},
		{"long healthy stream outlives ttft and idle", slowServer{firstAfter: 50 * ms, n: 30, gap: 30 * ms}, time.Minute, 100 * ms, 100 * ms, "", ""},
		{"global cuts a healthy stream", slowServer{n: -1, gap: 20 * ms}, 300 * ms, 100 * ms, 100 * ms, "global", "died mid-stream"},
		{"global before any token", slowServer{firstAfter: time.Hour}, 200 * ms, 0, 0, "global", "died after headers, before any token"},
		{"http error", slowServer{status: 500}, time.Minute, 100 * ms, 100 * ms, "http", "status 500"},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := runWatched(t, c.srv, c.global, c.ttft, c.idle)
			if got := errorKindOf(m.Error); got != c.kind {
				t.Fatalf("error %v: kind %q, want %q", m.Error, got, c.kind)
			}
			if c.msg != "" && !strings.Contains(m.Error.Error(), c.msg) {
				t.Errorf("error %q, want it to contain %q", m.Error, c.msg)
			}
		})
	}
}

// A plain (non-streamed) response is bounded only by the global deadline.
func TestRequestDeadlinesSkipPlainResponses(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"done"}}],"usage":{"prompt_tokens":5,"completion_tokens":1}}`)
	}))
	defer ts.Close()
	p, err := newReplayPoster(fmt.Sprintf("dynamic/%s,type=openai,model=m", ts.URL), llm.APIKeys{OpenAI: "k"}, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := withRequestDeadlines(context.Background(), time.Minute, 100*time.Millisecond, 100*time.Millisecond)
	defer cancel()
	m := p.do(ctx, minReplayReq(false), strings.Repeat("x", 300), 1, "s1", "i1", 1, &autoState{stream: newCompletionStream(200)}, nil)
	if m.Error != nil {
		t.Errorf("plain response failed: %v", m.Error)
	}
}

func TestErrorKindReachesJSONLAndProgressLine(t *testing.T) {
	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "m", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	st := newTestAutoState(4)
	cfg := AutoBenchmarkConfig{Model: "dynamic/http://h:1/v1,type=openai,model=m"}
	var cold time.Duration
	for _, e := range []error{
		&timeoutError{kind: ErrorKindTTFT, msg: "ttft timeout 300s"},
		&timeoutError{kind: ErrorKindIdle, msg: "stream idle 300s after 9 tokens"},
		&timeoutError{kind: ErrorKindIdle, msg: "stream idle 300s after 2 tokens"},
		&timeoutError{kind: ErrorKindGlobal, msg: "request timeout (global) 1h0m0s"},
		fmt.Errorf("status 503: busy"),
		fmt.Errorf("connection reset"),
		nil,
	} {
		recordReplayRequest(cfg, st, rdw, RequestMetrics{Error: e, Response: "x"}, false, &cold)
	}
	rdw.close()
	files, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if len(files) != 1 {
		t.Fatalf("jsonl files %v", files)
	}
	f, err := os.Open(files[0])
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var kinds []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		var r struct {
			ErrorKind *string `json:"error_kind"`
		}
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		k := "-"
		if r.ErrorKind != nil {
			k = *r.ErrorKind
		}
		kinds = append(kinds, k)
	}
	if got := strings.Join(kinds, ","); got != "ttft,idle,idle,global,http,other,-" {
		t.Errorf("error_kind per row %s", got)
	}
	want := " (ttft=1 idle=2 global=1 http=1 other=1)"
	if got := st.errorKinds.String(); got != want {
		t.Errorf("counts %q, want %q", got, want)
	}
	line := renderModelOneLiner(&displaySnapshot{totalErrors: 6, errorKinds: st.errorKinds.String()})
	if !strings.Contains(line, "errors=6"+want) {
		t.Errorf("progress line %q lacks %q", line, "errors=6"+want)
	}
	var none errorKindCounts
	if none.String() != "" {
		t.Error("no error still printed counts")
	}
}
