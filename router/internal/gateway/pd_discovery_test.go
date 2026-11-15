package gateway_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/dialect/openai"
	"github.com/weka/kvlab/router/internal/gateway"
	"github.com/weka/kvlab/router/internal/mockvllm"
	"github.com/weka/kvlab/router/internal/pd"
	"github.com/weka/kvlab/router/internal/pool"
	"github.com/weka/kvlab/router/internal/proxy"
	"github.com/weka/kvlab/router/internal/registry"
	"github.com/weka/kvlab/router/internal/routing"
)

// A discovery-backed pool: the prefill set is named up front by URL, the
// registry starts empty and endpoints arrive through ReconcileDiscovered, which
// is what the k8s discoverer calls.
type pdDiscovered struct {
	srv  *httptest.Server
	reg  *registry.Registry
	urls []string // one mock engine per index
	hits []int    // POSTs seen per engine
}

func newPDDiscovered(t *testing.T, n int, prefill ...int) *pdDiscovered {
	t.Helper()
	f := &pdDiscovered{reg: registry.New(registry.Options{}), hits: make([]int, n)}
	var pURLs []string
	for i := range n {
		eng := mockvllm.NewEngine(mockvllm.Config{DefaultMaxTokens: 4})
		t.Cleanup(eng.Close)
		h := mockvllm.NewServer(eng).Handler()
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				f.hits[i]++
			}
			h.ServeHTTP(w, r)
		}))
		t.Cleanup(ts.Close)
		f.urls = append(f.urls, ts.URL)
	}
	for _, i := range prefill {
		pURLs = append(pURLs, f.urls[i])
	}
	p := &pool.Pool{Name: "default", Registry: f.reg, Flow: mustFlow(t, affinityCfg()),
		PD: pd.New("default", pURLs), PDFlow: mustFlow(t, affinityCfg())}
	tbl, err := routing.NewTable([]routing.Rule{{Pool: p}})
	if err != nil {
		t.Fatal(err)
	}
	gw := gateway.New(gateway.Config{MaxBodyBytes: 64 << 20, DefaultCapacity: 1}, tbl,
		proxy.New(proxy.Config{MaxAttempts: 2}), openai.New())
	f.srv = httptest.NewServer(gw)
	t.Cleanup(f.srv.Close)
	return f
}

// live makes exactly the engines in idx the discovered, healthy endpoints.
func (f *pdDiscovered) live(t *testing.T, idx ...int) {
	t.Helper()
	var specs []registry.Spec
	for _, i := range idx {
		specs = append(specs, registry.Spec{URL: f.urls[i], Capacity: 1})
	}
	if _, err := f.reg.ReconcileDiscovered(specs); err != nil {
		t.Fatal(err)
	}
	for _, b := range f.reg.Snapshot().Backends {
		b.SetHealth(registry.Healthy)
	}
}

func (f *pdDiscovered) post(t *testing.T) (int, string) {
	t.Helper()
	c := http.Client{Timeout: 10 * time.Second} // a hang fails the test, not the run
	resp, err := c.Post(f.srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody("hello", false)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func (f *pdDiscovered) delta(before []int) []int {
	out := make([]int, len(f.hits))
	for i := range out {
		out[i] = f.hits[i] - before[i]
	}
	return out
}

// Engine 0 is the named prefill node, engines 1 and 2 are decode nodes. Roles
// follow the URL as endpoints appear: nothing live is a prompt 503, a decode
// node alone serves without a prefill leg, and once the prefill node appears
// requests run both legs.
func TestPDDiscoveredEndpointsArriveAfterStart(t *testing.T) {
	f := newPDDiscovered(t, 3, 0)

	if status, _ := f.post(t); status != http.StatusServiceUnavailable {
		t.Fatalf("empty pool: %d, want 503", status)
	}

	f.live(t, 1)
	before := append([]int(nil), f.hits...)
	if status, body := f.post(t); status != http.StatusOK {
		t.Fatalf("decode only: %d %s", status, body)
	}
	if d := f.delta(before); d[0] != 0 || d[1] != 1 {
		t.Errorf("decode only: hits %v, want the one D node served once and no prefill leg", d)
	}

	f.live(t, 0, 1, 2)
	before = append([]int(nil), f.hits...)
	if status, body := f.post(t); status != http.StatusOK {
		t.Fatalf("with prefill: %d %s", status, body)
	}
	d := f.delta(before)
	if d[0] != 1 || d[1]+d[2] != 1 {
		t.Errorf("with prefill: hits %v, want one prefill leg on node 0 and one decode leg on a D node", d)
	}
}

// A prefill endpoint that leaves costs the request its prefill leg, not its
// answer, and is used again when it returns.
func TestPDDiscoveredPrefillLeavesAndReturns(t *testing.T) {
	f := newPDDiscovered(t, 2, 0)
	f.live(t, 0, 1)
	legs := func() (p, d int) {
		before := append([]int(nil), f.hits...)
		if status, body := f.post(t); status != http.StatusOK {
			t.Fatalf("status %d: %s", status, body)
		}
		x := f.delta(before)
		return x[0], x[1]
	}
	if p, d := legs(); p != 1 || d != 1 {
		t.Fatalf("both live: P=%d D=%d, want 1/1", p, d)
	}
	f.live(t, 1)
	if p, d := legs(); p != 0 || d != 1 {
		t.Errorf("prefill gone: P=%d D=%d, want 0/1", p, d)
	}
	f.live(t, 0, 1)
	if p, d := legs(); p != 1 || d != 1 {
		t.Errorf("prefill back: P=%d D=%d, want 1/1", p, d)
	}
}

// With only prefill nodes live nothing can serve the client. It is told so at
// once, in words that name what is missing, rather than left waiting.
func TestPDNoDecodeEndpointFailsPromptly(t *testing.T) {
	f := newPDDiscovered(t, 2, 0)
	f.live(t, 0)
	start := time.Now()
	status, body := f.post(t)
	if status != http.StatusServiceUnavailable {
		t.Fatalf("status %d: %s", status, body)
	}
	if !strings.Contains(body, "decode") {
		t.Errorf("message does not name the missing role: %s", body)
	}
	if f.hits[0] != 0 {
		t.Errorf("a prefill leg ran with no decode node to follow it")
	}
	if time.Since(start) > 3*time.Second {
		t.Errorf("took %v", time.Since(start))
	}
}
