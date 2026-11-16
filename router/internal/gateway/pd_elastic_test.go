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

// These run the real mock engine behind the real gateway with an elastic P
// set (--pd-elastic): node 0 is the floor, the rest start as D nodes, and the
// manager converts and releases them as the prefill legs' load demands.

type elasticFleet struct {
	srv     *httptest.Server
	set     *pd.Set
	mgr     *prefill.Manager
	backend []*registry.Backend

	mu sync.Mutex
	// joined is when each node last joined the P set (zero while decode).
	joined    map[string]time.Time
	pLegsOn   map[int]int
	dLegsOn   map[int]int
	violation []string
	dNoParams atomic.Int64
}

const elasticGrace = 50 * time.Millisecond

func newElasticFleet(t *testing.T, n int, cfg prefill.Config, eng mockvllm.Config) *elasticFleet {
	t.Helper()
	f := &elasticFleet{joined: map[string]time.Time{}, pLegsOn: map[int]int{}, dLegsOn: map[int]int{}}
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
				f.mu.Lock()
				if c.isPrefillLeg() {
					f.pLegsOn[node]++
				} else {
					f.dLegsOn[node]++
					if _, has := body["kv_transfer_params"]; !has {
						f.dNoParams.Add(1)
					}
					// A D leg chosen just before its node joined the P set is
					// one of the decodes a converting node finishes; one that
					// arrives well after is a placement on a P node.
					if j := f.joined[url]; !j.IsZero() && time.Since(j) > elasticGrace {
						f.violation = append(f.violation, fmt.Sprintf("D leg on node %d, %v after it joined the P set", node, time.Since(j)))
					}
				}
				f.mu.Unlock()
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
	cfg.PoolName = t.Name()
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
	f.set = pd.New(t.Name(), floor)
	f.set.Elastic = f.mgr
	p := &pool.Pool{Name: t.Name(), Registry: reg, Flow: dflow, PD: f.set, PDFlow: pflow}
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

func (f *elasticFleet) inP() (n int) {
	for _, b := range f.backend {
		if f.set.IsPrefill(b) {
			n++
		}
	}
	return n
}

func (f *elasticFleet) post(t *testing.T, content string) {
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

func (f *elasticFleet) checkNoDecodeOnP(t *testing.T) {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, v := range f.violation {
		t.Error(v)
	}
	if n := f.dNoParams.Load(); n != 0 {
		t.Errorf("%d decode legs went without kv_transfer_params (prefill fell back)", n)
	}
}

func TestPDElasticMinimumServesPrefillOnConvertedNodeAndNoDecodeThere(t *testing.T) {
	f := newElasticFleet(t, 4, prefill.Config{MinPrefill: 2, MaxFleetFraction: 0.5,
		NodeBudgetTokens: 1}, mockvllm.Config{DefaultMaxTokens: 2})
	f.mgr.Step(f.backend)
	if n := f.inP(); n != 2 {
		t.Fatalf("P set = %d, want the minimum 2", n)
	}
	// Concurrent, so the floor is busy and legs go to the converted node too.
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
	f.mu.Lock()
	dLegs, pLegs := maps.Clone(f.dLegsOn), maps.Clone(f.pLegsOn)
	f.mu.Unlock()
	for node, n := range dLegs {
		if f.set.IsPrefill(f.backend[node]) {
			t.Errorf("%d D legs on P node %d", n, node)
		}
	}
	converted := 0
	for node, n := range pLegs {
		if !f.set.IsPrefill(f.backend[node]) {
			t.Errorf("%d P legs on D node %d", n, node)
		}
		if node != 0 && n > 0 {
			converted++
		}
	}
	if converted != 1 {
		t.Errorf("P legs on %d converted nodes, want the 1 converted to reach the minimum (by node: %v)", converted, pLegs)
	}
	f.checkNoDecodeOnP(t)
}

// The workbench scenario: a quiet start served by the floor alone, a late
// surge that holds the P set over budget and converts decode nodes up to the
// ceiling, and an idle period after which they all return to decode.
func TestPDElasticSurgeConvertsAndIdleReleases(t *testing.T) {
	const nodes = 6
	f := newElasticFleet(t, nodes, prefill.Config{
		NodeBudgetTokens: 30_000, // about one leg at a time per P node
		MaxFleetFraction: 0.5,    // at most 3 P nodes
		ConvertAfter:     150 * time.Millisecond,
		ReleaseIdle:      400 * time.Millisecond,
		ReleaseBelow:     0.5,
	}, mockvllm.Config{DefaultMaxTokens: 2, ColdInputTPS: 400_000, OutputTPS: 2_000})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go f.mgr.Run(ctx, func() []*registry.Backend { return f.backend })

	prompt := func(i int) string { return fmt.Sprintf("session %d %s", i, strings.Repeat("lorem ipsum ", 7_000)) }

	// Quiet: one at a time, the floor keeps up.
	for i := range 5 {
		f.post(t, prompt(i))
	}
	if n := f.inP(); n != 1 {
		t.Fatalf("P set = %d after quiet traffic, want the floor alone", n)
	}

	// Surge: far more concurrent prefill than the floor's budget.
	var wg sync.WaitGroup
	peak := atomic.Int64{}
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
	for i := range 60 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f.post(t, prompt(100+i))
		}()
	}
	wg.Wait()
	close(stop)
	conv := testutil.ToFloat64(metrics.PDConversions.WithLabelValues(t.Name(), "over_budget"))
	if conv < 1 || peak.Load() < 2 {
		t.Fatalf("surge converted %v nodes (peak P set %d), want conversions", conv, peak.Load())
	}
	if p := peak.Load(); p > nodes/2 {
		t.Errorf("peak P set %d over the ceiling %d", p, nodes/2)
	}
	f.mu.Lock()
	onConverted := 0
	for node, n := range f.pLegsOn {
		if node != 0 {
			onConverted += n
		}
	}
	f.mu.Unlock()
	if onConverted == 0 {
		t.Error("no prefill leg ran on a converted node")
	}

	// Idle: every converted node returns to decode.
	deadline := time.Now().Add(5 * time.Second)
	for f.inP() > 1 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if n := f.inP(); n != 1 {
		t.Fatalf("P set = %d after the idle period, want the floor alone", n)
	}
	rel := testutil.ToFloat64(metrics.PDReleases.WithLabelValues(t.Name(), "idle")) +
		testutil.ToFloat64(metrics.PDReleases.WithLabelValues(t.Name(), "low_water"))
	if rel != conv {
		t.Errorf("releases %v, want one per conversion (%v)", rel, conv)
	}
	// The released nodes are decode candidates again.
	if got := len(f.set.DecodeNodes(f.backend)); got != nodes-1 {
		t.Errorf("decode candidates = %d after the release, want %d", got, nodes-1)
	}
	for i := range 5 {
		f.post(t, prompt(1000+i))
	}
	f.checkNoDecodeOnP(t)
	f.mu.Lock()
	t.Logf("conversions %v, releases %v, peak P set %d, prefill legs by node %v", conv, rel, peak.Load(), f.pLegsOn)
	f.mu.Unlock()
}
