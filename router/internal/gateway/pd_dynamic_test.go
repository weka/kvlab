package gateway_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/weka/kvlab/router/internal/dialect/openai"
	"github.com/weka/kvlab/router/internal/gateway"
	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/mockvllm"
	"github.com/weka/kvlab/router/internal/pd"
	"github.com/weka/kvlab/router/internal/policy/affinity"
	"github.com/weka/kvlab/router/internal/pool"
	"github.com/weka/kvlab/router/internal/prefill"
	"github.com/weka/kvlab/router/internal/proxy"
	"github.com/weka/kvlab/router/internal/registry"
	"github.com/weka/kvlab/router/internal/routing"
)

// These run the real mock engine behind the real gateway in dynamic vanilla
// P/D (--prefill-split --prefill-split-mode=pd): node 0 is the floor, the rest
// start as decode nodes, and the prefill split's manager drains them into
// prefill and releases them as the prefill legs' load demands.

type dynFleet struct {
	srv     *httptest.Server
	pool    string // unique per fleet, so metric reads are this fleet's alone
	set     *pd.Set
	mgr     *prefill.Manager
	backend []*registry.Backend
	// failP makes P legs answer 500.
	failP atomic.Bool

	mu sync.Mutex
	// joined is when each node last left decode (zero while decode).
	joined    map[string]time.Time
	pLegsOn   map[int]int
	dLegsOn   map[int]int
	legs      map[string][]string // request id -> "P"/"D" in arrival order
	violation []string
	dNoParams int
}

// dynGrace is how long after a node leaves decode a D leg may still arrive
// there: one chosen just before the conversion is one of the decodes the
// drain waits for.
const dynGrace = 50 * time.Millisecond

func newDynFleet(t *testing.T, n int, cfg prefill.Config, eng mockvllm.Config) *dynFleet {
	t.Helper()
	f := &dynFleet{pool: fmt.Sprintf("%s-%d", t.Name(), time.Now().UnixNano()), joined: map[string]time.Time{}, pLegsOn: map[int]int{}, dLegsOn: map[int]int{}, legs: map[string][]string{}}
	reg := registry.New(registry.Options{})
	var floor []string
	for i := range n {
		e := mockvllm.NewEngine(eng)
		t.Cleanup(e.Close)
		ms := mockvllm.NewServer(e)
		ms.EngineID = fmt.Sprintf("engine-%d", i)
		h := ms.Handler()
		node := i
		var url string
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				raw, _ := io.ReadAll(r.Body)
				r.Body = io.NopCloser(bytes.NewReader(raw))
				var body map[string]json.RawMessage
				_ = json.Unmarshal(raw, &body)
				c := pdCall{node: node, body: body}
				id := r.Header.Get("X-Request-Id")
				f.mu.Lock()
				if c.isPrefillLeg() {
					f.pLegsOn[node]++
					f.legs[id] = append(f.legs[id], "P")
				} else {
					f.dLegsOn[node]++
					f.legs[id] = append(f.legs[id], "D")
					if _, has := body["kv_transfer_params"]; !has {
						f.dNoParams++
					}
					if j := f.joined[url]; !j.IsZero() && time.Since(j) > dynGrace {
						f.violation = append(f.violation, fmt.Sprintf("D leg on node %d, %v after it left decode", node, time.Since(j)))
					}
				}
				fail := c.isPrefillLeg() && f.failP.Load()
				f.mu.Unlock()
				if fail {
					http.Error(w, "boom", http.StatusInternalServerError)
					return
				}
			}
			h.ServeHTTP(w, r)
		}))
		t.Cleanup(ts.Close)
		url = ts.URL
		b, err := reg.Add(registry.Spec{URL: ts.URL, Prov: registry.ProvStatic, Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
		f.backend = append(f.backend, b)
		if i == 0 {
			floor = append(floor, ts.URL)
			f.joined[ts.URL] = time.Now().Add(-time.Hour)
		}
	}
	flow := affinity.Config{CostRouting: true}
	dflow, pflow := mustFlow(t, flow), mustFlow(t, flow)
	cfg.PD, cfg.Floor, cfg.CostPick = true, floor, true
	cfg.PoolName = f.pool
	cfg.OnModeChange = func(b *registry.Backend, from, to prefill.Mode) {
		// Called under the manager's lock; f.mu is never held while calling
		// into the manager.
		f.mu.Lock()
		defer f.mu.Unlock()
		switch {
		case from == prefill.Decode:
			f.joined[b.URL] = time.Now()
			dflow.Forget(b)
		case to == prefill.Decode:
			delete(f.joined, b.URL)
			pflow.Forget(b)
		}
	}
	f.mgr = prefill.New(cfg)
	f.set = pd.New(f.pool, floor)
	f.set.Dynamic = f.mgr
	p := &pool.Pool{Name: f.pool, Registry: reg, Flow: dflow, PD: f.set, PDFlow: pflow}
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

func (f *dynFleet) inP() (n int) {
	for _, b := range f.backend {
		if f.set.IsPrefill(b) {
			n++
		}
	}
	return n
}

func (f *dynFleet) post(t *testing.T, content string) {
	t.Helper()
	resp, err := http.Post(f.srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody(content, false)))
	if err != nil {
		t.Error(err)
		return
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status %d: %s", resp.StatusCode, b)
	}
}

// check asserts every request ran a P leg then a D leg, carrying params when
// wantParams, and that no D leg landed on a node out of decode.
func (f *dynFleet) check(t *testing.T, wantParams bool) {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, v := range f.violation {
		t.Error(v)
	}
	for id, seq := range f.legs {
		if strings.Join(seq, "") != "PD" {
			t.Errorf("request %s legs %v, want a P leg then a D leg", id, seq)
		}
	}
	if wantParams && f.dNoParams != 0 {
		t.Errorf("%d D legs without kv_transfer_params", f.dNoParams)
	}
}

func TestPDDynamicEveryRequestRunsAPLegThenADLegWithItsParams(t *testing.T) {
	f := newDynFleet(t, 4, prefill.Config{NodeBudgetTokens: 1, MaxFleetFraction: 0.5},
		mockvllm.Config{DefaultMaxTokens: 2})
	for batch := range 4 {
		var wg sync.WaitGroup
		for i := range 5 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				f.post(t, fmt.Sprintf("conversation %d-%d %s", batch, i, strings.Repeat("x", 4000)))
			}()
		}
		wg.Wait()
	}
	f.check(t, true)
	f.mu.Lock()
	n := len(f.legs)
	dLegs := maps.Clone(f.dLegsOn)
	f.mu.Unlock()
	if n != 20 {
		t.Errorf("%d requests seen, want 20", n)
	}
	if c := dLegs[0]; c != 0 {
		t.Errorf("%d D legs on the floor node", c)
	}
	if !f.set.IsPrefill(f.backend[0]) {
		t.Error("the floor node left the P set")
	}
	if ok := testutil.ToFloat64(metrics.PrefillJobs.WithLabelValues(f.pool, "ok", "")); ok != 20 {
		t.Errorf("router_prefill_jobs_total{outcome=ok} = %v, want 20 (one per request)", ok)
	}
}

func TestPDDynamicFailedPLegFallsBackToDecodeAlone(t *testing.T) {
	f := newDynFleet(t, 3, prefill.Config{}, mockvllm.Config{DefaultMaxTokens: 2})
	f.failP.Store(true)
	for i := range 3 {
		f.post(t, fmt.Sprintf("q%d %s", i, strings.Repeat("y", 2000)))
	}
	f.check(t, false)
	f.mu.Lock()
	noParams := f.dNoParams
	f.mu.Unlock()
	if noParams != 3 {
		t.Errorf("%d D legs went without params, want all 3 (recompute)", noParams)
	}
	if got := testutil.ToFloat64(metrics.PrefillJobs.WithLabelValues(f.pool, "failed", "status_5xx")); got != 3 {
		t.Errorf("router_prefill_jobs_total{failed,status_5xx} = %v, want 3", got)
	}
	if got := testutil.ToFloat64(metrics.PDFallbacks.WithLabelValues(f.pool, "prefill_failed")); got != 3 {
		t.Errorf("router_pd_fallbacks_total{prefill_failed} = %v, want 3", got)
	}
}

// The workbench scenario: a quiet start served by the floor alone, a late
// surge that holds the prefill nodes over budget and drains decode nodes into
// prefill up to the cap, and an idle period after which they return to decode.
func TestPDDynamicSurgeDrainsIntoPrefillAndIdleReleases(t *testing.T) {
	const nodes = 6
	f := newDynFleet(t, nodes, prefill.Config{
		NodeBudgetTokens: 30_000, // about one leg at a time per P node
		MaxFleetFraction: 0.5,    // at most 3 P nodes
		ConvertAfter:     150 * time.Millisecond,
		ReleaseIdle:      400 * time.Millisecond,
	}, mockvllm.Config{DefaultMaxTokens: 2, ColdInputTPS: 400_000, OutputTPS: 2_000})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go f.mgr.Run(ctx, func() []*registry.Backend { return f.backend })

	prompt := func(i int) string { return fmt.Sprintf("session %d %s", i, strings.Repeat("lorem ipsum ", 7_000)) }

	for i := range 5 { // quiet: the floor keeps up
		f.post(t, prompt(i))
	}
	if n := f.inP(); n != 1 {
		t.Fatalf("P set = %d after quiet traffic, want the floor alone", n)
	}

	// Surge: arrivals every 5ms, each leg ~50ms of prefill on one node.
	var wg sync.WaitGroup
	var peak atomic.Int64
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			case <-time.After(10 * time.Millisecond):
				if n := int64(f.inP()); n > peak.Load() {
					peak.Store(n)
				}
			}
		}
	}()
	for i := range 80 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f.post(t, prompt(100+i))
		}()
		time.Sleep(5 * time.Millisecond)
	}
	wg.Wait()
	close(stop)
	conv := testutil.ToFloat64(metrics.PrefillConversions.WithLabelValues(f.pool))
	if conv < 2 || peak.Load() != nodes/2 {
		t.Fatalf("surge drained %v nodes into prefill (peak P set %d), want the cap %d", conv, peak.Load(), nodes/2)
	}
	f.mu.Lock()
	onDrained := 0
	for node, n := range f.pLegsOn {
		if node != 0 {
			onDrained += n
		}
	}
	f.mu.Unlock()
	if onDrained == 0 {
		t.Error("no P leg ran on a drained node")
	}

	// Idle: every drained node returns to decode.
	deadline := time.Now().Add(5 * time.Second)
	for f.inP() > 1 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if n := f.inP(); n != 1 {
		t.Fatalf("P set = %d after the idle period, want the floor alone", n)
	}
	if rel := testutil.ToFloat64(metrics.PrefillReleases.WithLabelValues(f.pool)); rel != conv {
		t.Errorf("releases %v, want one per conversion (%v)", rel, conv)
	}
	if got := len(f.set.DecodeNodes(f.backend)); got != nodes-1 {
		t.Errorf("decode candidates = %d after the release, want %d", got, nodes-1)
	}
	for i := range 5 {
		f.post(t, prompt(1000+i))
	}
	f.check(t, true)
	f.mu.Lock()
	t.Logf("conversions %v, peak P set %d, P legs by node %v", conv, peak.Load(), f.pLegsOn)
	f.mu.Unlock()
}
