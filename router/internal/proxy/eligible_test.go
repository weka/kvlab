package proxy

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/weka/kvlab/router/internal/dialect/openai"
	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/registry"
)

// A selected backend that fails Eligible, asked while its lease is held, gets
// nothing sent to it: the request goes to the next candidate, and the
// abandoned pick is neither an attempt nor a verdict on the backend.
func TestServeRechecksEligibilityUnderTheLease(t *testing.T) {
	var hits [2]atomic.Int64
	var bs []*registry.Backend
	reg := registry.New(registry.Options{})
	for i := range 2 {
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			hits[i].Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[]}`))
		}))
		t.Cleanup(up.Close)
		b, err := reg.Add(registry.Spec{URL: up.URL})
		if err != nil {
			t.Fatal(err)
		}
		bs = append(bs, b)
	}
	var heldAtCheck int64
	rr := &policy.RoutingRequest{Eligible: func(b *registry.Backend) bool {
		if b == bs[0] {
			heldAtCheck = b.Inflight()
			return false
		}
		return true
	}}
	res := New(DefaultConfig()).Serve(httptest.NewRecorder(),
		httptest.NewRequest(http.MethodPost, "/v1/completions", nil), bs, firstSelector{},
		openai.New(), rr, []byte(`{}`), nil, nil, Auth{})
	if res.Err != nil || res.Backend != bs[1] || res.Attempts != 1 {
		t.Fatalf("served by %v in %d attempts, err %v; want the second backend in 1", res.Backend, res.Attempts, res.Err)
	}
	if heldAtCheck != 1 {
		t.Errorf("in-flight at the check = %d, want 1: the check must run under the lease", heldAtCheck)
	}
	if hits[0].Load() != 0 || hits[1].Load() != 1 {
		t.Errorf("upstream hits = %d, %d; want 0, 1", hits[0].Load(), hits[1].Load())
	}
	if bs[0].Inflight() != 0 || bs[0].Failed.Load() != 0 || bs[0].Served.Load() != 0 {
		t.Errorf("abandoned backend: in-flight %d failed %d served %d, want all 0",
			bs[0].Inflight(), bs[0].Failed.Load(), bs[0].Served.Load())
	}
}
