package prefill

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/registry"
)

// pdModeFleet runs the manager as TTFT-driven dynamic P/D with bs[0] as its
// initial prefill node: window 4, high 5s, low 2s, 5s between actions.
func pdModeFleet(t *testing.T, n int, cfg Config) *fleet {
	t.Helper()
	f := newFleet(t, n, Config{})
	cfg.PD = true
	cfg.Initial = []string{f.bs[0].URL + "/"} // normalised like the CLI's
	if cfg.MinPrefill == 0 {
		cfg.MinPrefill = 1
	}
	if cfg.TTFTWindow == 0 {
		cfg.TTFTWindow, cfg.TTFTHigh, cfg.TTFTLow = 4, 5*time.Second, 2*time.Second
	}
	if cfg.DrainMinInterval == 0 {
		cfg.DrainMinInterval = 5 * time.Second
	}
	if cfg.UnknownCostPer1K == 0 {
		cfg.UnknownCostPer1K = 10 * time.Millisecond
	}
	cfg.Clock, cfg.PoolName = f.clk, t.Name()
	cfg.PDMetrics = metrics.ForPD(t.Name())
	f.m = New(cfg)
	return f
}

// ttft feeds the window n samples of d.
func (f *fleet) ttft(n int, d time.Duration) {
	for range n {
		f.m.ObserveTTFT(d)
	}
}

func names(bs []*registry.Backend) []string {
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = b.URL
	}
	return out
}

func (f *fleet) setMode(b *registry.Backend, mode Mode, convertedAt time.Time) {
	f.m.mu.Lock()
	n := f.m.nodeLocked(b)
	n.mode, n.convertedAt = mode, convertedAt
	f.m.mu.Unlock()
}

func TestPDModeInitialNodeIsPrefillBeforeItIsSeenAndNeverADecodeCandidate(t *testing.T) {
	f := pdModeFleet(t, 3, Config{})
	if !f.m.IsPrefill(f.bs[0]) {
		t.Fatal("initial node is not prefill before the manager met it")
	}
	if got := f.m.DecodeNodes(f.bs); len(got) != 2 || got[0] == f.bs[0] || got[1] == f.bs[0] {
		t.Fatalf("decode candidates = %v, want the two other nodes", names(got))
	}
	// Only the prefill node is healthy: no decode leg goes there anyway.
	if got := f.m.DecodeNodes(f.bs[:1]); len(got) != 0 {
		t.Fatalf("decode candidates = %v, want none", names(got))
	}
}

// High TTFT drains one decode node; it takes no leg until its decodes end.
func TestPDModeHighTTFTDrainsOneNodeWhichTakesLegsOnlyOncePromoted(t *testing.T) {
	f := pdModeFleet(t, 4, Config{MaxFleetFraction: 0.75})
	for _, b := range f.bs[2:] {
		defer lease.Acquire(b).Release() // older requests than node 1's
	}
	time.Sleep(2 * time.Millisecond)
	decoding := lease.Acquire(f.bs[1])
	f.ttft(4, 8*time.Second)
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[1]) != Draining {
		t.Fatalf("modes %v, want node 1 (youngest oldest request) draining", f.modes())
	}
	if got := f.m.ActivePrefill(f.bs); len(got) != 1 || got[0] != f.bs[0] {
		t.Errorf("active prefill = %v, want the initial node only while node 1 drains", names(got))
	}
	if n := count(f.modes(), Draining) + count(f.modes(), Prefill); n != 2 {
		t.Errorf("one window drove %d nodes out of decode, want 2 (initial + one drain)", n)
	}
	decoding.Release()
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[1]) != Prefill || len(f.m.ActivePrefill(f.bs)) != 2 {
		t.Errorf("modes %v after the drain, want node 1 an active prefill node", f.modes())
	}
}

// After an action the next waits for the interval AND a refreshed window.
func TestPDModeActionsWaitForTheIntervalAndAFreshWindow(t *testing.T) {
	f := pdModeFleet(t, 6, Config{MaxFleetFraction: 0.8})
	f.ttft(4, 8*time.Second)
	f.m.Tick(f.bs) // drain 1
	f.clk.Advance(10 * time.Second)
	f.ttft(3, 8*time.Second) // only 3 of 4 fresh
	f.m.Tick(f.bs)
	if got := f.m.State().Conversions; got != 1 {
		t.Fatalf("conversions = %d on a window not yet refreshed, want 1", got)
	}
	f.ttft(1, 8*time.Second)
	f.clk.Advance(time.Second) // the window is fresh, but stale data counts after 10s only
	f.m.Tick(f.bs)
	if got := f.m.State().Conversions; got != 2 {
		t.Fatalf("conversions = %d after the window refreshed, want 2", got)
	}
	f.ttft(4, 8*time.Second)
	f.clk.Advance(4 * time.Second)
	f.m.Tick(f.bs)
	if got := f.m.State().Conversions; got != 2 {
		t.Errorf("conversions = %d 4s after the last action, want 2 (interval 5s)", got)
	}
}

// Between the thresholds nothing happens.
func TestPDModeHoldsBetweenTheThresholds(t *testing.T) {
	f := pdModeFleet(t, 4, Config{MaxFleetFraction: 0.75})
	f.setMode(f.bs[1], Prefill, f.clk.Now())
	f.ttft(4, 3*time.Second)
	f.m.Tick(f.bs)
	if c, r := f.m.State().Conversions, f.m.State().Releases; c != 0 || r != 0 {
		t.Errorf("conversions %d releases %d at a 3s mean, want none", c, r)
	}
}

// Low TTFT abandons the newest drain first, one per action.
func TestPDModeLowTTFTAbandonsTheNewestDrainFirst(t *testing.T) {
	f := pdModeFleet(t, 5, Config{MaxFleetFraction: 0.8})
	for _, b := range f.bs[1:] {
		defer lease.Acquire(b).Release() // still decoding: drains stay draining
	}
	f.setMode(f.bs[1], Draining, f.clk.Now().Add(time.Second))
	f.setMode(f.bs[2], Draining, f.clk.Now().Add(2*time.Second))
	f.ttft(4, 500*time.Millisecond)
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[2]) != Decode || f.m.ModeOf(f.bs[1]) != Draining {
		t.Fatalf("modes %v, want the newest drain (node 2) abandoned first", f.modes())
	}
	f.clk.Advance(5 * time.Second)
	f.ttft(4, 500*time.Millisecond)
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[1]) != Decode {
		t.Errorf("modes %v, want node 1 abandoned on the next action", f.modes())
	}
}

// Low TTFT returns the prefill node holding the fewest prefix tokens: it takes
// no new leg, finishes the ones in flight, then becomes a decode node.
func TestPDModeLowTTFTReleasesTheNodeHoldingLeastAfterItsLegs(t *testing.T) {
	held := map[string]int64{}
	f := pdModeFleet(t, 4, Config{MaxFleetFraction: 0.75, PrefixValues: func() map[string]int64 { return held }})
	f.setMode(f.bs[1], Prefill, f.clk.Now().Add(time.Second))
	f.setMode(f.bs[2], Prefill, f.clk.Now().Add(2*time.Second))
	held[f.bs[0].URL], held[f.bs[1].URL], held[f.bs[2].URL] = 50_000, 1_000, 90_000
	f.m.LegStart(f.bs[1]) // node 1 has a leg in flight
	f.ttft(4, time.Second)
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[1]) != Prefill {
		t.Fatal("released with a leg still in flight")
	}
	for _, b := range f.m.ActivePrefill(f.bs) {
		if b == f.bs[1] {
			t.Fatal("the node returning to decode is still offered legs")
		}
	}
	f.m.LegDone(f.bs[1], 300*time.Millisecond, true)
	if f.m.ModeOf(f.bs[1]) != Decode {
		t.Errorf("modes %v, want node 1 (least prefix held) back in decode once its leg ended", f.modes())
	}
}

// No recent request takes no action: a router up before its clients keeps
// its prefill nodes. Fast prefills shrink the set to the minimum, one node per
// action, and no further.
func TestPDModeIdleHoldsAndFastPrefillsShrinkToTheMinimum(t *testing.T) {
	f := pdModeFleet(t, 5, Config{MaxFleetFraction: 0.8, MinPrefill: 2})
	for i, b := range f.bs[1:4] {
		f.setMode(b, Prefill, f.clk.Now().Add(time.Duration(i+1)*time.Second))
	}
	for range 6 {
		f.m.Tick(f.bs)
		f.clk.Advance(5 * time.Second)
	}
	if n := count(f.modes(), Prefill); n != 4 {
		t.Fatalf("%d prefill nodes after an idle half-minute, want all 4 kept (modes %v)", n, f.modes())
	}
	for range 6 {
		f.ttft(4, 100*time.Millisecond)
		f.m.Tick(f.bs)
		f.clk.Advance(5 * time.Second)
	}
	if n := count(f.modes(), Prefill); n != 2 {
		t.Errorf("%d prefill nodes after fast prefills, want the minimum 2 (modes %v)", n, f.modes())
	}
}

// The minimum is reached by draining, at start when no node is named.
func TestPDModeMinimumIsReachedByDraining(t *testing.T) {
	f := newFleet(t, 5, Config{})
	f.m = New(Config{PD: true, MinPrefill: 2, MaxFleetFraction: 0.6, TTFTWindow: 4,
		TTFTHigh: 5 * time.Second, TTFTLow: 2 * time.Second, Clock: f.clk, PoolName: t.Name()})
	for _, b := range f.bs {
		defer lease.Acquire(b).Release()
	}
	f.m.Tick(f.bs)
	f.m.Tick(f.bs)
	if n := count(f.modes(), Draining) + count(f.modes(), Prefill); n != 2 {
		t.Errorf("%d nodes out of decode, want the minimum 2 (modes %v)", n, f.modes())
	}
}

// A leg goes to the cheapest active node: estimated leg time plus the cost of
// the prompt tokens the node does not hold. Known prefix is free.
func TestPDModePickWeighsEstimatedTimeAgainstUnknownTokens(t *testing.T) {
	f := pdModeFleet(t, 3, Config{UnknownCostPer1K: 10 * time.Millisecond, MaxFleetFraction: 0.7})
	f.setMode(f.bs[1], Prefill, f.clk.Now())
	active := f.m.ActivePrefill(f.bs)
	// Both idle, no service history: node 1 holds 80k of a 100k prompt.
	if got := f.m.PickPrefill(active, 100_000, []int64{0, 80_000}); got != f.bs[1] {
		t.Fatalf("picked %s, want the holder", got.URL)
	}
	// The holder is slow: 1s legs with 4 in flight is 2s est; the other
	// node is 0.3s. 0.8s of unknown tokens does not pay for 1.7s.
	f.m.LegDone(f.bs[1], time.Second, true) // seed its EWMA
	for range 4 {
		f.m.LegStart(f.bs[1])
	}
	f.m.LegDone(f.bs[0], 300*time.Millisecond, true)
	if got := f.m.PickPrefill(active, 100_000, []int64{0, 80_000}); got != f.bs[0] {
		t.Errorf("picked %s, want the faster node once the holder's backlog outweighs the prefix", got.URL)
	}
}

// A draining node only ever finishes its decodes; under the reject mode it
// also takes no job while a prefill node exists, and its drain is abandoned
// only once the overflow clears.
func TestRejectModeDrainedNodeTakesNoJobUntilItIsAPrefillNode(t *testing.T) {
	f := newFleet(t, 3, Config{MaxFleetFraction: 0.7})
	f.setMode(f.bs[0], Prefill, time.Time{})
	defer lease.Acquire(f.bs[2]).Release()
	defer lease.Acquire(f.bs[2]).Release()
	time.Sleep(2 * time.Millisecond)
	decode := lease.Acquire(f.bs[1])
	full, _ := f.m.Reserve(f.bs, DefaultNodeBudgetTokens, nil)
	_ = f.m.WaitReady(context.Background(), full)
	j, err := f.m.Reserve(f.bs, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if f.m.ModeOf(f.bs[1]) != Draining || j.Backend != f.bs[0] {
		t.Fatalf("modes %v, job on %s; want node 1 draining and the job on the prefill node", f.modes(), j.Backend.URL)
	}
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[1]) != Draining {
		t.Errorf("drain abandoned while the overflow lasts (mode %v)", f.m.ModeOf(f.bs[1]))
	}
	decode.Release()
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[1]) != Prefill || j.Backend != f.bs[1] {
		t.Errorf("after the drain node 1 is %v and the job on %s; want prefill and the job moved there",
			f.m.ModeOf(f.bs[1]), j.Backend.URL)
	}
	if err := f.m.WaitReady(context.Background(), j); err != nil {
		t.Fatal(err)
	}
	j.Done()
	full.Done()
}

func TestJobChargeAndWaitAreObserved(t *testing.T) {
	f := newFleet(t, 3, Config{MaxFleetFraction: 0.7})
	tokens0 := histogram(t, metrics.PrefillJobTokens.WithLabelValues(t.Name()))
	wait0 := histogram(t, metrics.PrefillJobWait.WithLabelValues(t.Name()))
	j, _ := f.m.Reserve(f.bs, 70, nil)
	f.clk.Advance(300 * time.Millisecond)
	_ = f.m.WaitReady(context.Background(), j)
	j.Done()
	tokens := histogram(t, metrics.PrefillJobTokens.WithLabelValues(t.Name()))
	wait := histogram(t, metrics.PrefillJobWait.WithLabelValues(t.Name()))
	if n, sum := tokens.GetSampleCount()-tokens0.GetSampleCount(), tokens.GetSampleSum()-tokens0.GetSampleSum(); n != 1 || sum != 70 {
		t.Errorf("job tokens count %d sum %v, want 1 and 70", n, sum)
	}
	if n := wait.GetSampleCount() - wait0.GetSampleCount(); n != 1 {
		t.Errorf("job wait count %d, want 1", n)
	}
}

func histogram(t *testing.T, o prometheus.Observer) *dto.Histogram {
	t.Helper()
	var m dto.Metric
	if err := o.(prometheus.Metric).Write(&m); err != nil {
		t.Fatal(err)
	}
	return m.GetHistogram()
}

func TestPDModeProvisioningIsLoggedAndCounted(t *testing.T) {
	var buf bytes.Buffer
	f := pdModeFleet(t, 4, Config{MaxFleetFraction: 0.75, Log: slog.New(slog.NewTextHandler(&buf, nil))})
	f.ttft(4, 8*time.Second)
	f.m.Tick(f.bs)
	out := buf.String()
	for _, want := range []string{`msg="decode node draining for prefill"`, "reason=ttft_high", "prefill_finish_mean=8s", "prefill_finish_samples=4"} {
		if !strings.Contains(out, want) {
			t.Errorf("log lacks %q:\n%s", want, out)
		}
	}
	var m dto.Metric
	_ = metrics.PDProvisionActions.WithLabelValues(t.Name(), "drain").Write(&m)
	if m.GetCounter().GetValue() < 1 {
		t.Error("drain action not counted")
	}
}

// timeFleet runs the time window: the last 30s, high 1.5s, low 0.5s.
func timeFleet(t *testing.T, n int, cfg Config) *fleet {
	cfg.TTFTWindow, cfg.TTFTWindowTime = 32, 30*time.Second
	cfg.TTFTHigh, cfg.TTFTLow = 1500*time.Millisecond, 500*time.Millisecond
	return pdModeFleet(t, n, cfg)
}

// spread feeds n samples of d over the next span, ending now+span.
func (f *fleet) spread(n int, d, span time.Duration) {
	for range n {
		f.clk.Advance(span / time.Duration(n))
		f.m.ObserveTTFT(d)
	}
}

// The time window averages only the samples of the last span.
func TestPDModeTimeWindowAveragesTheLastSpanOnly(t *testing.T) {
	f := timeFleet(t, 4, Config{MaxFleetFraction: 0.75})
	f.spread(40, 4*time.Second, 30*time.Second) // old and slow
	f.spread(40, time.Second, 30*time.Second)   // the last 30s: 1s
	f.clk.Advance(time.Millisecond)             // the last old sample is now just outside
	f.m.mu.Lock()
	mean, k := f.m.windowMeanLocked(f.clk.Now())
	f.m.mu.Unlock()
	if mean != time.Second || k != 40 {
		t.Errorf("window mean %v over %d samples, want 1s over the last 30s' 40", mean, k)
	}
	f.m.Tick(f.bs)
	if f.m.State().Conversions != 0 {
		t.Error("drained on samples older than the window")
	}
}

// After an action the next waits for a full window of samples taken after it.
func TestPDModeTimeWindowNeedsAFullFreshWindowAfterAnAction(t *testing.T) {
	f := timeFleet(t, 6, Config{MaxFleetFraction: 0.8})
	f.spread(40, 3*time.Second, 30*time.Second)
	f.m.Tick(f.bs) // drain 1
	if got := f.m.State().Conversions; got != 1 {
		t.Fatalf("conversions = %d on a slow full window, want 1", got)
	}
	f.spread(40, 3*time.Second, 29*time.Second) // still slow, but 29s since
	f.m.Tick(f.bs)
	if got := f.m.State().Conversions; got != 1 {
		t.Fatalf("conversions = %d 29s after the last action, want 1 (window 30s)", got)
	}
	f.spread(2, 3*time.Second, time.Second)
	f.m.Tick(f.bs)
	if got := f.m.State().Conversions; got != 2 {
		t.Errorf("conversions = %d after a full fresh window, want 2", got)
	}
}

// A window with a few requests holds, and so does an empty one.
func TestPDModeTimeWindowThinOrEmptyHolds(t *testing.T) {
	f := timeFleet(t, 4, Config{MaxFleetFraction: 0.75})
	f.setMode(f.bs[1], Prefill, f.clk.Now().Add(time.Second))
	f.spread(10, 3*time.Second, 30*time.Second) // 10 < 16
	f.m.Tick(f.bs)
	if c, r := f.m.State().Conversions, f.m.State().Releases; c != 0 || r != 0 {
		t.Fatalf("acted on 10 samples (conversions %d releases %d), want hold", c, r)
	}
	f.clk.Advance(31 * time.Second) // nothing in the window now
	f.m.Tick(f.bs)
	if c, r := f.m.State().Conversions, f.m.State().Releases; c != 0 || r != 0 || f.m.ModeOf(f.bs[1]) != Prefill {
		t.Errorf("acted on an empty window (conversions %d releases %d, modes %v), want hold", c, r, f.modes())
	}
}
