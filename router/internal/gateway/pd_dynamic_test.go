package gateway_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"

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
	pflow   *affinity.Policy
	dflow   *affinity.Policy
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
	cfg.PD, cfg.Initial = true, floor
	if cfg.MinPrefill == 0 {
		cfg.MinPrefill = 1
	}
	if cfg.TTFTWindow == 0 {
		// Thresholds no plain test reaches: they never provision.
		cfg.TTFTWindow, cfg.TTFTHigh, cfg.TTFTLow = 8, time.Hour, time.Nanosecond
	}
	if cfg.UnknownCostPer1K == 0 {
		cfg.UnknownCostPer1K = 10 * time.Millisecond
	}
	cfg.PrefixValues = pflow.BackendTokens
	cfg.PoolName = f.pool
	f.set = pd.New(f.pool, floor)
	cfg.PDMetrics = f.set.M
	cfg.OnModeChange = func(b *registry.Backend, from, to prefill.Mode) {
		// Called under the manager's lock; f.mu is never held while calling
		// into the manager.
		f.mu.Lock()
		defer f.mu.Unlock()
		switch {
		case from == prefill.Decode:
			f.joined[b.URL] = time.Now()
		case to == prefill.Decode:
			delete(f.joined, b.URL)
			pflow.Forget(b)
		}
		if to == prefill.Prefill {
			dflow.Forget(b) // as the pool does: decode marks go at promotion
		}
	}
	f.mgr = prefill.New(cfg)
	f.pflow, f.dflow = pflow, dflow
	f.set.Dynamic = f.mgr
	stored, err := pd.NewStored(flow)
	if err != nil {
		t.Fatal(err)
	}
	f.set.Stored = stored
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
	_ = dLegs // D legs on a node while it was in the P set fail f.check
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

func (f *dynFleet) postMessages(t *testing.T, msgs []map[string]any) {
	t.Helper()
	b, _ := json.Marshal(map[string]any{"model": "m", "max_tokens": 2, "messages": msgs})
	resp, err := http.Post(f.srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Error(err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Errorf("status %d: %s", resp.StatusCode, body)
	}
}

// A leg is charged what its P node must compute, not what it can load: a
// prompt the fleet prefilled before is in the shared tier, so once the P
// tree has forgotten it (its tail TTL, or the node's release) the D tree's
// record still credits it, and the next turn is charged its new tokens only.
func TestPDDynamicLegChargeCreditsThePrefixTheTierHolds(t *testing.T) {
	f := newDynFleet(t, 3, prefill.Config{MaxFleetFraction: 0.5},
		mockvllm.Config{DefaultMaxTokens: 2, BaseLatency: 400 * time.Millisecond})
	history := []map[string]any{{"role": "user", "content": "shared " + strings.Repeat("context ", 5_000)}}
	f.postMessages(t, history)
	// Both trees lose the prefix, as on the P node's release and the D
	// node's promotion; the tier still holds it, and so does the record.
	for _, b := range f.backend {
		f.pflow.Forget(b)
		f.dflow.Forget(b)
	}

	turn := append(history,
		map[string]any{"role": "assistant", "content": "ok"},
		map[string]any{"role": "user", "content": "next question " + strings.Repeat("why ", 50)})
	before := pdHistogram(t, metrics.PrefillJobTokens.WithLabelValues(f.pool))
	f.postMessages(t, turn)
	after := pdHistogram(t, metrics.PrefillJobTokens.WithLabelValues(f.pool))
	charged := after.GetSampleSum() - before.GetSampleSum()
	if after.GetSampleCount()-before.GetSampleCount() != 1 || charged <= 0 || charged > 1_000 {
		t.Errorf("turn 2 charged %v tokens, want only its new tokens (~100), not the ~10k prompt", charged)
	}
}

func TestPDLegChargeIsThePromptLessWhatIsHeldOrStored(t *testing.T) {
	for _, c := range []struct{ total, local, stored, want int64 }{
		{10_000, 0, 0, 10_000},      // nothing known: the whole prompt
		{10_000, 9_000, 0, 1_000},   // P tree holds it
		{10_000, 0, 8_192, 1_808},   // tier holds it, cut to the store unit
		{10_000, 9_500, 8_192, 500}, // the deeper credit wins
		{10_000, 10_000, 0, 1},      // at least one token
	} {
		if got := gateway.PDLegCharge(c.total, c.local, c.stored); got != c.want {
			t.Errorf("charge(%d, local %d, stored %d) = %d, want %d", c.total, c.local, c.stored, got, c.want)
		}
	}
}

// A followed conversation's legs are logged with their charge breakdown and
// how far each prompt matches the previous leg; the held-depth histograms
// see every leg.
func TestPDDynamicLegChargeDiagnostics(t *testing.T) {
	var conv string
	for i := 0; ; i++ {
		if c, ok := gateway.PDDiagConversation(fmt.Sprintf("wk-s%d-t1-ab12", i)); ok {
			conv = c
			break
		}
	}
	var buf syncBuffer
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	defer slog.SetDefault(old)

	f := newDynFleet(t, 3, prefill.Config{MaxFleetFraction: 0.5}, mockvllm.Config{DefaultMaxTokens: 2})
	history := []map[string]any{{"role": "user", "content": strings.Repeat("context ", 3_000)}}
	send := func(turn int, msgs []map[string]any) {
		b, _ := json.Marshal(map[string]any{"model": "m", "max_tokens": 2, "messages": msgs})
		req, _ := http.NewRequest(http.MethodPost, f.srv.URL+"/v1/chat/completions", bytes.NewReader(b))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Request-Id", fmt.Sprintf("%s-t%d-ab12", conv, turn))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}
	send(1, history)
	send(2, append(history,
		map[string]any{"role": "assistant", "content": "ok"},
		map[string]any{"role": "user", "content": "next"}))

	var lines []string
	for _, l := range strings.Split(buf.String(), "\n") {
		if strings.Contains(l, `msg="pd leg charge"`) {
			lines = append(lines, l)
		}
	}
	if len(lines) != 2 {
		t.Fatalf("%d charge lines for the followed conversation, want 2:\n%s", len(lines), buf.String())
	}
	if !strings.Contains(lines[0], "diverge_unit=-1") {
		t.Errorf("first leg: %s", lines[0])
	}
	// The second prompt extends the first: it matches every unit of it.
	var prevUnits int
	for _, kv := range strings.Fields(lines[1]) {
		if strings.HasPrefix(kv, "prev_units=") {
			_, _ = fmt.Sscanf(kv, "prev_units=%d", &prevUnits)
		}
	}
	if prevUnits == 0 || !strings.Contains(lines[1], fmt.Sprintf("diverge_unit=%d ", prevUnits)) {
		t.Errorf("second leg does not extend the first: %s", lines[1])
	}
	for _, want := range []string{"total_tokens=", "prefill_tree_tokens=", "decode_tree_tokens=", "stored_tokens=", "charge_tokens=", "conversation=" + conv} {
		if !strings.Contains(lines[1], want) {
			t.Errorf("second leg lacks %s: %s", want, lines[1])
		}
	}
	for _, tree := range []string{"prefill", "decode"} {
		var m dto.Metric
		_ = metrics.PDPrefillHeldTokens.WithLabelValues(f.pool, tree).(prometheus.Metric).Write(&m)
		if got := m.GetHistogram().GetSampleCount(); got != 2 {
			t.Errorf("held tokens{tree=%s} observed %d legs, want 2", tree, got)
		}
	}
}

func pdHistogram(t *testing.T, o prometheus.Observer) *dto.Histogram {
	t.Helper()
	var m dto.Metric
	if err := o.(prometheus.Metric).Write(&m); err != nil {
		t.Fatal(err)
	}
	return m.GetHistogram()
}

// The ramp from one prefill node on the real gateway: a surge of long prompts
// on the single prefill node pushes the mean prefill-finish time over the high
// threshold and drains decode nodes into prefill, up to the cap; legs never
// wait in the router. Light traffic afterwards (fast legs) shrinks the prefill
// set back to the minimum of one; silence alone would not.
func TestPDDynamicTTFTRampFromOneExpandsUnderLoadAndShrinksUnderLightTraffic(t *testing.T) {
	const nodes = 6
	f := newDynFleet(t, nodes, prefill.Config{
		MaxFleetFraction: 0.5, // at most 3 prefill nodes
		TTFTWindow:       8,
		TTFTHigh:         300 * time.Millisecond,
		TTFTLow:          100 * time.Millisecond,
		DrainMinInterval: 100 * time.Millisecond,
	}, mockvllm.Config{DefaultMaxTokens: 2, ColdInputTPS: 400_000, OutputTPS: 2_000})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go f.mgr.Run(ctx, func() []*registry.Backend { return f.backend })
	prompt := func(i int) string { return fmt.Sprintf("session %d %s", i, strings.Repeat("lorem ipsum ", 7_000)) }

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
	for i := range 150 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f.post(t, prompt(i))
		}()
		time.Sleep(4 * time.Millisecond)
	}
	wg.Wait()
	close(stop)
	conv := testutil.ToFloat64(metrics.PrefillConversions.WithLabelValues(f.pool))
	if conv < 1 || peak.Load() < 2 || peak.Load() > nodes/2 {
		t.Fatalf("surge drained %v nodes (peak P set %d), want a ramp from 1 within the cap %d", conv, peak.Load(), nodes/2)
	}
	time.Sleep(500 * time.Millisecond)
	if n := f.inP(); n < 2 {
		t.Fatalf("P set = %d after half a second of silence, want the expansion kept (no data, no action)", n)
	}
	deadline := time.Now().Add(15 * time.Second)
	for i := 0; f.inP() > 1 && time.Now().Before(deadline); i++ {
		f.post(t, fmt.Sprintf("light %d", i)) // short prompts: fast prefill legs
	}
	if n := f.inP(); n != 1 {
		t.Fatalf("P set = %d after light traffic, want the minimum of one", n)
	}
	f.check(t, true)
	if got := pdCounter(t, f.pool, "drain"); got < 1 {
		t.Errorf("drain actions counted %v, want at least 1", got)
	}
	t.Logf("conversions %v, peak P set %d", conv, peak.Load())
}

func pdCounter(t *testing.T, pool, action string) float64 {
	t.Helper()
	var sum float64
	for _, ar := range metrics.PDProvisionActionReasons {
		if ar[0] == action {
			sum += testutil.ToFloat64(metrics.PDProvisionActions.WithLabelValues(pool, ar[0], ar[1]))
		}
	}
	return sum
}

// The provisioning signal is the time to the prefill's finish: arrival to the
// P leg's kv_transfer_params. Time the decode leg adds is not in it.
func TestPDDynamicSignalStopsAtThePrefillHandOff(t *testing.T) {
	f := newDynFleet(t, 3, prefill.Config{MaxFleetFraction: 0.5},
		mockvllm.Config{DefaultMaxTokens: 2, BaseLatency: 200 * time.Millisecond})
	start := time.Now()
	f.post(t, "hello "+strings.Repeat("x", 2000))
	total := time.Since(start)
	f.mgr.Tick(f.backend)
	mean := testutil.ToFloat64(metrics.PDTTFTWindowMean.WithLabelValues(f.pool))
	if total < 400*time.Millisecond {
		t.Fatalf("setup: the request took %v, want both legs' 200ms", total)
	}
	if mean < 0.2 || mean > 0.35 {
		t.Errorf("signal = %.3fs for a %v request, want the P leg's ~0.2s, not the decode leg's time too", mean, total)
	}
}
