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
	// resends counts bypassed requests that are not jobs: the decode resend,
	// or the fallback to the refusing backend.
	resends atomic.Int64
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
	if doc.MaxTokens != 1 {
		j.resends.Add(1)
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
	mgr  *prefill.Manager
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
	f.mgr = prefill.New(prefill.Config{PoolName: f.pool, NodeConcurrency: 1})
	p := &pool.Pool{Name: "default", Registry: reg, Flow: mustFlow(t, affinityCfg()), Split: f.mgr}
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

func (f *jobFailFleet) cancelled(reason string) float64 {
	return testutil.ToFloat64(metrics.PrefillJobs.WithLabelValues(f.pool, "cancelled", reason))
}

// assertSettled checks that every job's slot, budget share and pending tokens
// came back, and that nothing was counted as failed.
func (f *jobFailFleet) assertSettled(t *testing.T) {
	t.Helper()
	waitFor(t, func() bool {
		for _, n := range f.mgr.State().Nodes {
			if n.PendingTokens != 0 || n.RunningJobs != 0 || n.QueuedJobs != 0 || n.RunningTokens != 0 {
				return false
			}
		}
		return true
	})
	for _, r := range metrics.PrefillJobFailureReasons {
		if n := f.failed(r); n != 0 {
			t.Errorf("a cancellation counted as failed{reason=%q}", r)
		}
	}
	if st := f.mgr.State(); st.JobsFailed != 0 || st.BypassFallbacks != 0 {
		t.Errorf("jobs_failed %d, bypass fallbacks %d; want 0 for a caller that left", st.JobsFailed, st.BypassFallbacks)
	}
}

// A caller that leaves while its job runs cancels the job. That is counted as
// cancelled, not failed; nothing more is sent for the request; and the job's
// slot, budget share and pending tokens come back.
func TestPrefillJobCancelledWhileRunningSendsNothingMore(t *testing.T) {
	lb := captureLog(t)
	f := newJobFailFleet(t, 0)
	f.be.jobDelay.Store(int64(2 * time.Second))
	f.post(t, 100*time.Millisecond)

	waitFor(t, func() bool { return f.cancelled("running") == 1 })
	f.assertSettled(t)
	time.Sleep(50 * time.Millisecond) // a resend would have arrived by now
	if n := f.be.resends.Load(); n != 0 {
		t.Errorf("%d requests sent with the bypass after the caller left", n)
	}
	if l := lb.lines(`msg="prefill job cancelled"`); len(l) != 1 || !strings.Contains(l[0], "reason=running") {
		t.Errorf("cancel log lines = %q", l)
	}
}

// A caller that leaves while its job waits for a slot: the job never runs, is
// counted as cancelled while waiting, and nothing is sent for it.
func TestPrefillJobCancelledWhileWaitingSendsNothing(t *testing.T) {
	lb := captureLog(t)
	f := newJobFailFleet(t, 0)
	f.be.jobDelay.Store(int64(500 * time.Millisecond))
	first := make(chan struct{})
	go func() { defer close(first); f.post(t, 0) }() // holds the one slot
	time.Sleep(100 * time.Millisecond)
	f.post(t, 100*time.Millisecond) // queues behind it, then leaves
	<-first

	waitFor(t, func() bool { return f.cancelled("waiting") == 1 })
	if n := testutil.ToFloat64(metrics.PrefillJobs.WithLabelValues(f.pool, "ok", "")); n != 1 {
		t.Errorf("ok jobs = %v, want the first", n)
	}
	f.assertSettled(t)
	if n := f.be.resends.Load(); n != 1 {
		t.Errorf("%d bypassed resends, want only the first request's", n)
	}
	if l := lb.lines("reason=waiting"); len(l) != 1 {
		t.Errorf("waiting-cancel log lines = %q", l)
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
