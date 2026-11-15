package gateway_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/weka/kvlab/router/internal/clock"
	"github.com/weka/kvlab/router/internal/dialect/openai"
	"github.com/weka/kvlab/router/internal/gateway"
	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/pool"
	"github.com/weka/kvlab/router/internal/prefill"
	"github.com/weka/kvlab/router/internal/proxy"
	"github.com/weka/kvlab/router/internal/registry"
	"github.com/weka/kvlab/router/internal/routing"
)

// jobBackend refuses every request without the bypass as a heavy prefill and
// answers prefill jobs (max_tokens=1) after jobDelay with jobStatus.
type jobBackend struct {
	jobDelay  atomic.Int64 // nanoseconds
	jobStatus atomic.Int32
}

func (j *jobBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		return
	}
	body, _ := io.ReadAll(r.Body)
	var doc struct {
		MaxTokens float64        `json:"max_tokens"`
		XArgs     map[string]any `json:"vllm_xargs"`
	}
	_ = json.Unmarshal(body, &doc)
	w.Header().Set("Content-Type", "application/json")
	if doc.XArgs["weka_prefill"] == nil {
		w.Header().Set(prefill.RejectHeader, "9000")
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}
	if doc.MaxTokens == 1 {
		select {
		case <-time.After(time.Duration(j.jobDelay.Load())):
		case <-r.Context().Done():
			return
		}
		if st := int(j.jobStatus.Load()); st != 0 {
			w.WriteHeader(st)
			return
		}
	}
	_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
}

// logBuffer captures the default logger's output for one test.
type logBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *logBuffer) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func (l *logBuffer) lines(substr string) (out []string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, line := range strings.Split(l.buf.String(), "\n") {
		if strings.Contains(line, substr) {
			out = append(out, line)
		}
	}
	return out
}

func captureLog(t *testing.T) *logBuffer {
	t.Helper()
	lb := &logBuffer{}
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(lb, nil)))
	t.Cleanup(func() { slog.SetDefault(old) })
	return lb
}

var jobFailFleets atomic.Int64

type jobFailFleet struct {
	srv  *httptest.Server
	be   *jobBackend
	clk  *clock.Fake
	pool string
}

// newJobFailFleet is two backends, one of which the first job converts; the
// prefill node runs one job at a time.
func newJobFailFleet(t *testing.T, requestTimeout time.Duration) *jobFailFleet {
	t.Helper()
	// A pool name of its own, so the counters start at zero on -count reruns.
	f := &jobFailFleet{be: &jobBackend{}, clk: clock.NewFake(time.Now()),
		pool: fmt.Sprintf("%s#%d", t.Name(), jobFailFleets.Add(1))}
	reg := registry.New(registry.Options{})
	for range 2 {
		ts := httptest.NewServer(f.be)
		t.Cleanup(ts.Close)
		b, err := reg.Add(registry.Spec{URL: ts.URL, Prov: registry.ProvStatic, Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
	}
	p := &pool.Pool{Name: "default", Registry: reg, Flow: mustFlow(t, affinityCfg()),
		Split: prefill.New(prefill.Config{PoolName: f.pool, NodeConcurrency: 1})}
	tbl, _ := routing.NewTable([]routing.Rule{{Pool: p}})
	f.srv = httptest.NewServer(gateway.New(
		gateway.Config{MaxBodyBytes: 1 << 20, DefaultCapacity: 1, Clock: f.clk}, tbl,
		proxy.New(proxy.Config{MaxAttempts: 2, RequestTimeout: requestTimeout}), openai.New()))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *jobFailFleet) post(t *testing.T, timeout time.Duration) {
	t.Helper()
	ctx := context.Background()
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, f.srv.URL+"/v1/chat/completions",
		strings.NewReader(chatBody("hi", false)))
	req.Header.Set("Content-Type", "application/json")
	if resp, err := http.DefaultClient.Do(req); err == nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}
}

func (f *jobFailFleet) failed(reason string) float64 {
	return testutil.ToFloat64(metrics.PrefillJobs.WithLabelValues(f.pool, "failed", reason))
}

// Every failed prefill job is counted under its reason and logged with it and
// the backend.
func TestPrefillJobFailureIsCountedAndLoggedWithItsReason(t *testing.T) {
	for _, c := range []struct {
		reason         string
		requestTimeout time.Duration
		jobDelay       time.Duration
		jobStatus      int
		clientTimeout  time.Duration
		want           []string
	}{
		{reason: "timeout", requestTimeout: 100 * time.Millisecond, jobDelay: 2 * time.Second},
		{reason: "status_5xx", jobStatus: http.StatusInternalServerError, want: []string{"status=500"}},
		{reason: "status_4xx", jobStatus: http.StatusBadRequest, want: []string{"status=400"}},
		{reason: "client_cancelled", jobDelay: 2 * time.Second, clientTimeout: 100 * time.Millisecond},
	} {
		t.Run(c.reason, func(t *testing.T) {
			lb := captureLog(t)
			f := newJobFailFleet(t, c.requestTimeout)
			f.be.jobDelay.Store(int64(c.jobDelay))
			f.be.jobStatus.Store(int32(c.jobStatus))
			f.post(t, c.clientTimeout)

			waitFor(t, func() bool { return f.failed(c.reason) == 1 })
			for _, r := range metrics.PrefillJobFailureReasons {
				if r != c.reason && f.failed(r) != 0 {
					t.Errorf("also counted as %s", r)
				}
			}
			waitFor(t, func() bool { return len(lb.lines("prefill job failed")) == 1 })
			line := lb.lines("prefill job failed")[0]
			for _, w := range append([]string{"reason=" + c.reason, "backend=http://", "tokens=9000"}, c.want...) {
				if !strings.Contains(line, w) {
					t.Errorf("log line lacks %q: %s", w, line)
				}
			}
		})
	}
}

// A job whose caller leaves while it waits for a slot never ran: it is
// counted as abandoned.
func TestPrefillJobAbandonedWhileWaitingIsCounted(t *testing.T) {
	lb := captureLog(t)
	f := newJobFailFleet(t, 0)
	f.be.jobDelay.Store(int64(500 * time.Millisecond))
	first := make(chan struct{})
	go func() { defer close(first); f.post(t, 0) }() // holds the one slot
	time.Sleep(100 * time.Millisecond)
	f.post(t, 100*time.Millisecond) // queues behind it, then leaves
	<-first
	waitFor(t, func() bool { return f.failed("abandoned") == 1 })
	if n := testutil.ToFloat64(metrics.PrefillJobs.WithLabelValues(f.pool, "ok", "")); n != 1 {
		t.Errorf("ok jobs = %v, want the first", n)
	}
	if l := lb.lines("reason=abandoned"); len(l) != 1 {
		t.Errorf("abandoned log lines = %q", l)
	}
}

// Failures of one reason log at most once per interval; the next line says how
// many it stands for.
func TestPrefillJobFailureLogIsRateLimitedPerReason(t *testing.T) {
	lb := captureLog(t)
	f := newJobFailFleet(t, 0)
	f.be.jobStatus.Store(http.StatusInternalServerError)
	for range 5 {
		f.post(t, 0)
	}
	if n := f.failed("status_5xx"); n != 5 {
		t.Fatalf("counted %v failures, want 5", n)
	}
	if l := lb.lines("prefill job failed"); len(l) != 1 {
		t.Fatalf("%d lines for 5 failures inside one interval:\n%s", len(l), strings.Join(l, "\n"))
	}
	f.clk.Advance(11 * time.Second)
	f.post(t, 0)
	l := lb.lines("prefill job failed")
	if len(l) != 2 || !strings.Contains(l[1], "suppressed_since_last=4") {
		t.Errorf("after the interval:\n%s\nwant a second line with suppressed_since_last=4", strings.Join(l, "\n"))
	}
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	for end := time.Now().Add(5 * time.Second); time.Now().Before(end); time.Sleep(5 * time.Millisecond) {
		if cond() {
			return
		}
	}
	t.Fatal("condition not reached")
}

// The router's request timeout bounds a job's time on the engine, from when
// it is sent: time spent waiting in the router for a slot does not count.
func TestPrefillJobTimeoutStartsWhenTheJobIsSent(t *testing.T) {
	f := newJobFailFleet(t, 400*time.Millisecond)
	f.be.jobDelay.Store(int64(300 * time.Millisecond))
	var wg sync.WaitGroup
	for range 2 { // the second waits ~300ms for the slot, then runs 300ms
		wg.Add(1)
		go func() { defer wg.Done(); f.post(t, 0) }()
		time.Sleep(20 * time.Millisecond)
	}
	wg.Wait()
	if n := testutil.ToFloat64(metrics.PrefillJobs.WithLabelValues(f.pool, "ok", "")); n != 2 {
		t.Errorf("ok jobs = %v, want 2; timeouts = %v", n, f.failed("timeout"))
	}
}
