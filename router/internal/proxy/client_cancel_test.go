package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/circuit"
	"github.com/weka/kvlab/router/internal/dialect/openai"
	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/registry"
)

// firstSelector picks the first candidate, so a retry lands on the next one.
type firstSelector struct{}

func (firstSelector) Select(_ context.Context, c []*registry.Backend, _ *policy.RoutingRequest) (*registry.Backend, error) {
	return c[0], nil
}
func (firstSelector) Name() string { return "first" }

// A client that hangs up before the backend has answered is neither a backend
// failure nor a reason to try another backend: the breakers stay closed, the
// failure counters stay at zero and the request is dispatched exactly once.
func TestClientCancelBeforeResponseIsNotAnUpstreamFailure(t *testing.T) {
	var hits atomic.Int64
	release := make(chan struct{})
	slow := func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}
	upA, upB := httptest.NewServer(http.HandlerFunc(slow)), httptest.NewServer(http.HandlerFunc(slow))
	defer upA.Close()
	defer upB.Close()
	defer close(release)
	a, b := decodeBackend(t, upA.URL), decodeBackend(t, upB.URL)

	const n = 60
	p := New(DefaultConfig())
	results := make([]Result, n)
	var wg sync.WaitGroup
	cancels := make([]context.CancelFunc, n)
	for i := 0; i < n; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		cancels[i] = cancel
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodPost, "/v1/completions", nil).WithContext(ctx)
			results[i] = p.Serve(httptest.NewRecorder(), req, []*registry.Backend{a, b},
				firstSelector{}, openai.New(), nil, []byte(`{"prompt":"x"}`), nil, nil, Auth{})
		}(i)
	}
	waitFor(t, "all requests to reach a backend", func() bool { return hits.Load() == n })
	for _, c := range cancels {
		c() // the load driver stops: every request is still waiting for headers
	}
	wg.Wait()

	if got := hits.Load(); got != n {
		t.Errorf("upstream dispatches = %d, want %d (a cancelled request must not be retried)", got, n)
	}
	for i, r := range results {
		if r.Attempts != 1 {
			t.Fatalf("request %d: attempts = %d, want 1", i, r.Attempts)
		}
	}
	for name, be := range map[string]*registry.Backend{"A": a, "B": b} {
		if s := be.CB.State(); s != circuit.Closed {
			t.Errorf("backend %s breaker = %v, want closed", name, s)
		}
		if f := be.Failed.Load(); f != 0 {
			t.Errorf("backend %s failures = %d, want 0", name, f)
		}
		if s := be.Served.Load(); s != 0 {
			t.Errorf("backend %s served = %d, want 0", name, s)
		}
	}
}

// The router's own request timeout still charges the backend: only the
// inbound context being cancelled exempts it.
func TestRequestTimeoutStillChargesBackend(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer up.Close()
	be := decodeBackend(t, up.URL)
	cfg := DefaultConfig()
	cfg.RequestTimeout = 50 * time.Millisecond
	cfg.MaxAttempts = 1
	p := New(cfg)
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(""))
	res := p.Serve(httptest.NewRecorder(), req, []*registry.Backend{be}, firstSelector{},
		openai.New(), nil, []byte(`{}`), nil, nil, Auth{})
	if res.Err == nil {
		t.Fatal("timeout must surface as an error")
	}
	if be.Failed.Load() != 1 {
		t.Errorf("failures = %d, want 1", be.Failed.Load())
	}
}
