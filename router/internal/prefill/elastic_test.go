package prefill

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/registry"
)

// pdFleet is a fleet whose manager runs the elastic P set with bs[0] as the
// floor, unless cfg names another.
func pdFleet(t *testing.T, n int, cfg Config) *fleet {
	t.Helper()
	cfg.PD = true
	f := newFleet(t, n, cfg)
	if cfg.Floor == nil {
		cfg.Floor = []string{f.bs[0].URL + "/"} // normalised like the CLI's
		cfg.Clock, cfg.PoolName = f.clk, t.Name()
		f.m = New(cfg)
	}
	return f
}

// run reserves a job and waits until it holds its slot.
func (f *fleet) run(t *testing.T, tokens int64) *Job {
	t.Helper()
	j, err := f.m.Reserve(f.bs, tokens, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.m.WaitReady(context.Background(), j); err != nil {
		t.Fatal(err)
	}
	return j
}

func (f *fleet) inP() (n int) {
	for _, b := range f.bs {
		if f.m.IsPrefill(b) {
			n++
		}
	}
	return n
}

func conversions(t *testing.T, reason string) float64 {
	return testutil.ToFloat64(metrics.PDConversions.WithLabelValues(t.Name(), reason))
}

func releases(t *testing.T, reason string) float64 {
	return testutil.ToFloat64(metrics.PDReleases.WithLabelValues(t.Name(), reason))
}

func TestPDFloorIsPrefillBeforeItIsSeenAndNeverDecode(t *testing.T) {
	f := pdFleet(t, 3, Config{})
	if !f.m.IsPrefill(f.bs[0]) {
		t.Fatal("floor node is not in the P set before the manager met it")
	}
	got := f.m.DecodeNodes(f.bs)
	if len(got) != 2 || got[0] == f.bs[0] || got[1] == f.bs[0] {
		t.Fatalf("decode candidates = %v, want the two non-floor nodes", urls(got))
	}
	// Idle forever: a floor node is never released.
	f.run(t, 10).Done()
	f.clk.Advance(time.Hour)
	f.m.Step(f.bs)
	if f.m.ModeOf(f.bs[0]) != Prefill {
		t.Errorf("floor node mode = %v after an idle hour, want prefill", f.m.ModeOf(f.bs[0]))
	}
}

func TestPDNeverReturnsAPNodeAsADecodeCandidate(t *testing.T) {
	f := pdFleet(t, 2, Config{})
	// Only the floor node is healthy: the prefill split would serve decode on
	// it rather than refuse; P/D must not.
	if got := f.m.DecodeNodes(f.bs[:1]); len(got) != 0 {
		t.Fatalf("decode candidates = %v, want none", urls(got))
	}
}

func TestPDMinConvertsTheLeastDecodeLoadedNodes(t *testing.T) {
	f := pdFleet(t, 5, Config{MinPrefill: 2, MaxFleetFraction: 0.8})
	// Decode load by cost routing's measure: node 3 carries the least context.
	for i, tok := range []int64{0, 9000, 5000, 100, 7000} {
		f.bs[i].AddActiveTokens(tok)
	}
	f.m.Step(f.bs)
	if !f.m.IsPrefill(f.bs[3]) {
		t.Fatalf("modes = %v, want node 3 in the P set", f.modes())
	}
	if n := f.inP(); n != 2 {
		t.Errorf("P set has %d nodes, want the minimum 2", n)
	}
	if got := conversions(t, "min"); got != 1 {
		t.Errorf("conversions{reason=min} = %v, want 1", got)
	}
	// The minimum is never released, however idle.
	f.clk.Advance(time.Hour)
	f.m.Step(f.bs)
	if n := f.inP(); n != 2 {
		t.Errorf("P set has %d nodes after an idle hour, want 2", n)
	}
}

func TestPDConvertingNodeTakesLegsAtOnceAndBecomesPrefillWhenItsDecodesEnd(t *testing.T) {
	f := pdFleet(t, 3, Config{MinPrefill: 2, MaxFleetFraction: 0.7})
	f.bs[2].AddActiveTokens(1) // node 1 is the conversion target
	decode := lease.Acquire(f.bs[1])
	f.m.Step(f.bs)
	if f.m.ModeOf(f.bs[1]) != Converting {
		t.Fatalf("modes = %v, want node 1 converting", f.modes())
	}
	// Fill the floor so the next leg must go to the converting node, which
	// still runs its decode: no drain wait.
	f.run(t, DefaultNodeBudgetTokens)
	done := make(chan *Job, 1)
	go func() { done <- f.run(t, 1000) }()
	select {
	case j := <-done:
		if j.Backend != f.bs[1] {
			t.Fatalf("leg ran on %s, want the converting node", j.Backend.URL)
		}
		defer j.Done()
		// The leg is in flight on the node, as the proxy would have it.
		leg := lease.Acquire(j.Backend)
		defer leg.Release()
	case <-time.After(2 * time.Second):
		t.Fatal("leg waited for the converting node's decode to finish")
	}
	f.m.Step(f.bs)
	if f.m.ModeOf(f.bs[1]) != Converting {
		t.Errorf("mode = %v while its decode runs, want converting", f.m.ModeOf(f.bs[1]))
	}
	decode.Release()
	f.m.Step(f.bs)
	if f.m.ModeOf(f.bs[1]) != Prefill {
		t.Errorf("mode = %v once its decode ended, want prefill", f.m.ModeOf(f.bs[1]))
	}
	if got := f.m.DecodeNodes(f.bs); len(got) != 1 || got[0] != f.bs[2] {
		t.Errorf("decode candidates = %v, want node 2 only", urls(got))
	}
}

func TestPDConvertsOnlyAfterTheSetStaysOverBudget(t *testing.T) {
	const budget = 100
	f := pdFleet(t, 4, Config{NodeBudgetTokens: budget, ConvertAfter: 10 * time.Second})
	busy := f.run(t, budget) // the floor is full
	queued, err := f.m.Reserve(f.bs, 50, nil)
	if err != nil {
		t.Fatal(err)
	}
	if queued.Backend != f.bs[0] {
		t.Fatalf("overflow leg placed on %s, want queued on the floor node", queued.Backend.URL)
	}
	got := make(chan error, 1)
	go func() { got <- f.m.WaitReady(context.Background(), queued) }()

	f.clk.Advance(9 * time.Second)
	f.m.Step(f.bs)
	if n := f.inP(); n != 1 {
		t.Fatalf("P set grew to %d before --prefill-convert-after", n)
	}
	f.clk.Advance(time.Second)
	f.m.Step(f.bs)
	if n := f.inP(); n != 2 {
		t.Fatalf("P set = %d after a sustained overflow, want 2 (modes %v)", n, f.modes())
	}
	if got := conversions(t, "over_budget"); got != 1 {
		t.Errorf("conversions{reason=over_budget} = %v, want 1", got)
	}
	select {
	case err := <-got:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("queued leg did not move to the converted node")
	}
	if queued.Backend == f.bs[0] {
		t.Error("queued leg still on the full floor node")
	}
	queued.Done()
	busy.Done()
}

func TestPDNewLegConvertsWhenTheOverflowIsOldEnough(t *testing.T) {
	const budget = 100
	f := pdFleet(t, 4, Config{NodeBudgetTokens: budget, ConvertAfter: 5 * time.Second})
	defer f.run(t, budget).Done()
	q, _ := f.m.Reserve(f.bs, 10, nil) // over budget from now
	defer q.Done()
	f.clk.Advance(5 * time.Second)
	j, err := f.m.Reserve(f.bs, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer j.Done()
	if j.Backend == f.bs[0] || f.m.ModeOf(j.Backend) != Converting {
		t.Errorf("leg on %s (%v), want a newly converting node", j.Backend.URL, f.m.ModeOf(j.Backend))
	}
}

func TestPDCeilingCapsTheSetAndKeepsADecodeNode(t *testing.T) {
	for _, c := range []struct {
		name string
		n    int
		frac float64
		want int
	}{
		{"half of four", 4, 0.5, 2},
		{"never the last decode node", 3, 1.0, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := pdFleet(t, c.n, Config{NodeBudgetTokens: 10, MaxFleetFraction: c.frac, MinPrefill: 10})
			var jobs []*Job
			for range 10 {
				j, err := f.m.Reserve(f.bs, 10, nil)
				if err != nil {
					t.Fatal(err)
				}
				jobs = append(jobs, j)
				f.clk.Advance(time.Minute)
				f.m.Step(f.bs)
			}
			if n := f.inP(); n != c.want {
				t.Errorf("P set = %d, want the cap %d", n, c.want)
			}
			if len(f.m.DecodeNodes(f.bs)) == 0 {
				t.Error("no decode node left")
			}
			for _, j := range jobs {
				j.Done()
			}
		})
	}
}

func TestPDReleasesAnIdleConvertedNodeAfterReleaseIdle(t *testing.T) {
	f := pdFleet(t, 4, Config{NodeBudgetTokens: 100, ReleaseIdle: 30 * time.Second})
	f.run(t, 100) // floor full, never finished
	j, err := f.m.Reserve(f.bs, 50, nil)
	if err != nil {
		t.Fatal(err)
	}
	f.m.Step(f.bs) // ConvertAfter 0: converts at once
	if f.m.WaitReady(context.Background(), j) != nil || j.Backend == f.bs[0] {
		t.Fatalf("leg not running on a converted node: %s", j.Backend.URL)
	}
	conv := j.Backend
	j.Done()
	f.clk.Advance(29 * time.Second)
	f.m.Step(f.bs)
	if !f.m.IsPrefill(conv) {
		t.Fatal("released before --prefill-release-idle")
	}
	f.clk.Advance(time.Second)
	f.m.Step(f.bs)
	if f.m.IsPrefill(conv) {
		t.Fatalf("still in the P set after %v idle (mode %v)", 30*time.Second, f.m.ModeOf(conv))
	}
	if got := releases(t, "idle"); got != 1 {
		t.Errorf("releases{reason=idle} = %v, want 1", got)
	}
}

// The low-water rule releases a converted node that cost routing keeps a
// little busy, so it is never idle, once the rest of the set can carry the
// load; and only after the load has stayed that low for the hold time.
func TestPDLowWaterReleaseWaitsForASustainedLowAndRestartsAfterEachRelease(t *testing.T) {
	const budget = 1000
	f := pdFleet(t, 5, Config{NodeBudgetTokens: budget, MaxFleetFraction: 0.8, MinPrefill: 3,
		ReleaseIdle: 20 * time.Second, ReleaseBelow: 0.5})
	f.m.Step(f.bs) // floor + 2 converted
	if n := f.inP(); n != 3 {
		t.Fatalf("P set = %d, want 3", n)
	}
	// Drop the minimum so the converted nodes may go.
	f.m.cfg.MinPrefill = 1

	// Load the whole set at 700 of 3000: above 50% of the 2000 the set would
	// have without one node (1000), so nothing is released.
	var jobs []*Job
	for range 7 {
		jobs = append(jobs, f.run(t, 100))
	}
	f.clk.Advance(time.Minute)
	f.m.Step(f.bs)
	if n := f.inP(); n != 3 {
		t.Fatalf("released under load above the watermark: P set = %d", n)
	}
	// 3 of them finish: 400 pending, under 1000. Every P node still has
	// work, so none is idle; the low-water hold starts now.
	for _, j := range jobs[:3] {
		j.Done()
	}
	jobs = jobs[3:]
	f.m.Step(f.bs)
	f.clk.Advance(19 * time.Second)
	f.m.Step(f.bs)
	if n := f.inP(); n != 3 {
		t.Fatalf("released before the low lasted --prefill-release-idle: P set = %d", n)
	}
	f.clk.Advance(time.Second)
	f.m.Step(f.bs)
	if n := f.inP(); n != 2 {
		t.Fatalf("P set = %d after a sustained low, want 2", n)
	}
	if got := releases(t, "low_water"); got != 1 {
		t.Errorf("releases{reason=low_water} = %v, want 1", got)
	}
	// The hold restarts: still low (400 <= 500 for one node left), but the
	// next release needs a full window of its own.
	f.clk.Advance(19 * time.Second)
	f.m.Step(f.bs)
	if n := f.inP(); n != 2 {
		t.Errorf("second release came %v after the first, before a full hold", 19*time.Second)
	}
	f.clk.Advance(time.Second)
	f.m.Step(f.bs)
	if n := f.inP(); n != 1 {
		t.Errorf("P set = %d, want the floor alone", n)
	}
	if f.m.ModeOf(f.bs[0]) != Prefill {
		t.Error("the floor node was released")
	}
	for _, j := range jobs {
		j.Done()
	}
}

// A load that rises again before the hold is over resets it.
func TestPDLowWaterHoldResetsWhenLoadRises(t *testing.T) {
	const budget = 1000
	f := pdFleet(t, 4, Config{NodeBudgetTokens: budget, MaxFleetFraction: 0.5, MinPrefill: 2,
		ReleaseIdle: 20 * time.Second, ReleaseBelow: 0.5})
	f.m.Step(f.bs)
	f.m.cfg.MinPrefill = 1
	small := f.run(t, 100)
	f.m.Step(f.bs) // low from now
	f.clk.Advance(15 * time.Second)
	big := f.run(t, 900) // 1000 > 500: no longer low
	f.m.Step(f.bs)
	big.Done()
	f.m.Step(f.bs) // low again, from now
	f.clk.Advance(15 * time.Second)
	f.m.Step(f.bs)
	if n := f.inP(); n != 2 {
		t.Errorf("P set = %d: the hold was not restarted when the load rose", n)
	}
	small.Done()
}

func TestPDNodeWithQueuedLegsIsNotReleased(t *testing.T) {
	f := pdFleet(t, 3, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.7, MinPrefill: 2,
		ReleaseIdle: time.Second, ReleaseBelow: 0.99})
	f.m.Step(f.bs)
	f.m.cfg.MinPrefill = 1
	var conv *registry.Backend
	for _, b := range f.bs[1:] {
		if f.m.IsPrefill(b) {
			conv = b
		}
	}
	f.m.mu.Lock()
	n := f.m.nodes[conv.URL]
	n.queue = append(n.queue, &Job{n: n, tokens: 1})
	f.m.mu.Unlock()
	f.clk.Advance(time.Minute)
	f.m.Step(f.bs)
	f.clk.Advance(time.Minute)
	f.m.Step(f.bs)
	if !f.m.IsPrefill(conv) {
		t.Error("a node with legs queued on it was released")
	}
}

func TestPDConversionAndReleaseAreLoggedWithReasonAndLoad(t *testing.T) {
	var buf bytes.Buffer
	log := slog.New(slog.NewTextHandler(&buf, nil))
	f := pdFleet(t, 4, Config{NodeBudgetTokens: 100, ReleaseIdle: time.Second, Log: log})
	f.run(t, 100)
	j, _ := f.m.Reserve(f.bs, 10, nil)
	_ = f.m.WaitReady(context.Background(), j)
	j.Done()
	f.clk.Advance(time.Second)
	f.m.Step(f.bs)
	out := buf.String()
	for _, want := range []string{
		`msg="decode node converted to prefill"`, "reason=over_budget", "pending_tokens=", "node_budget_tokens=100",
		`msg="prefill node released to decode"`, "reason=idle", "idle_for=1s",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("log lacks %q:\n%s", want, out)
		}
	}
}

func TestPDModeCountsAndQueueTokensArePublished(t *testing.T) {
	f := pdFleet(t, 4, Config{NodeBudgetTokens: 100, MinPrefill: 2})
	for _, b := range f.bs[1:] {
		defer lease.Acquire(b).Release() // decodes in flight: converting
	}
	f.m.Step(f.bs)
	gauge := func(mode string) float64 {
		return testutil.ToFloat64(metrics.PDPrefillNodes.WithLabelValues(t.Name(), mode))
	}
	if gauge("prefill") != 1 || gauge("converting") != 1 || gauge("decode") != 2 {
		t.Errorf("nodes prefill/converting/decode = %v/%v/%v, want 1/1/2",
			gauge("prefill"), gauge("converting"), gauge("decode"))
	}
	a := f.run(t, 100)
	b := f.run(t, 100)
	q, _ := f.m.Reserve(f.bs, 30, nil)
	tok := func(state string) float64 {
		return testutil.ToFloat64(metrics.PDPrefillQueueTokens.WithLabelValues(t.Name(), state))
	}
	if tok("running") != 200 || tok("queued") != 30 {
		t.Errorf("queue tokens running/queued = %v/%v, want 200/30", tok("running"), tok("queued"))
	}
	q.Done()
	a.Done()
	b.Done()
	if tok("running") != 0 || tok("queued") != 0 {
		t.Errorf("queue tokens running/queued = %v/%v after all finished, want 0/0", tok("running"), tok("queued"))
	}
}

func urls(bs []*registry.Backend) []string {
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = b.URL
	}
	return out
}
