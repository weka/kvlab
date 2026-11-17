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

// pdModeFleet runs the manager as dynamic P/D with bs[0] as the floor.
func pdModeFleet(t *testing.T, n int, cfg Config) *fleet {
	t.Helper()
	f := newFleet(t, n, Config{})
	cfg.PD = true
	cfg.Initial = []string{f.bs[0].URL + "/"} // normalised like the CLI's
	if cfg.MinPrefill == 0 {
		cfg.MinPrefill = 1 // keeps node 0 the one prefill node at rest
	}
	cfg.Clock, cfg.PoolName = f.clk, t.Name()
	f.m = New(cfg)
	return f
}

func names(bs []*registry.Backend) []string {
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = b.URL
	}
	return out
}

func TestPDModeFloorIsPrefillBeforeItIsSeenAndIsNeverReleased(t *testing.T) {
	for _, idle := range []time.Duration{0, time.Second} {
		f := pdModeFleet(t, 3, Config{ReleaseIdle: idle})
		if !f.m.IsPrefill(f.bs[0]) {
			t.Fatal("floor node is not prefill before the manager met it")
		}
		if got := f.m.DecodeNodes(f.bs); len(got) != 2 || got[0] == f.bs[0] || got[1] == f.bs[0] {
			t.Fatalf("decode candidates = %v, want the two non-floor nodes", names(got))
		}
		j, err := f.m.Reserve(f.bs, 10, nil)
		if err != nil || j.Backend != f.bs[0] {
			t.Fatalf("first leg on %v (%v), want the floor node", j, err)
		}
		if err := f.m.WaitReady(context.Background(), j); err != nil {
			t.Fatal(err)
		}
		j.Done()
		f.clk.Advance(time.Hour)
		f.m.releaseExpired()
		if f.m.ModeOf(f.bs[0]) != Prefill {
			t.Errorf("release-idle %v: floor node mode = %v after an idle hour, want prefill", idle, f.m.ModeOf(f.bs[0]))
		}
	}
}

func TestPDModeNeverReturnsAPrefillOrDrainingNodeForDecode(t *testing.T) {
	f := pdModeFleet(t, 3, Config{MaxFleetFraction: 0.7})
	// Only the floor node is healthy: the reject mode would serve the
	// request on it rather than refuse; P/D must not place a D leg there.
	if got := f.m.DecodeNodes(f.bs[:1]); len(got) != 0 {
		t.Fatalf("decode candidates = %v, want none", names(got))
	}
	// Fill the floor so the next leg drains a decode node.
	defer lease.Acquire(f.bs[1]).Release()
	defer lease.Acquire(f.bs[2]).Release()
	defer lease.Acquire(f.bs[2]).Release()
	j1, _ := f.m.Reserve(f.bs, DefaultNodeBudgetTokens, nil)
	defer j1.Done()
	j2, err := f.m.Reserve(f.bs, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer j2.Done()
	if f.m.ModeOf(f.bs[1]) != Draining {
		t.Fatalf("modes %v, want node 1 draining", f.modes())
	}
	if got := f.m.DecodeNodes(f.bs); len(got) != 1 || got[0] != f.bs[2] {
		t.Errorf("decode candidates = %v, want node 2 only", names(got))
	}
}

// A job never waits on a draining node while a prefill node exists: it
// queues on the prefill node, and moves to the drained node only once that
// node's decodes have finished and it is a prefill node.
func TestDrainedNodeTakesNoJobUntilItIsAPrefillNode(t *testing.T) {
	for _, pd := range []bool{false, true} {
		f := newFleet(t, 3, Config{MaxFleetFraction: 0.7})
		if pd {
			f = pdModeFleet(t, 3, Config{MaxFleetFraction: 0.7})
		} else {
			// The reject mode: node 0 is already a prefill node.
			f.m.mu.Lock()
			f.m.nodeLocked(f.bs[0]).mode = Prefill
			f.m.mu.Unlock()
		}
		decode := lease.Acquire(f.bs[1])
		defer lease.Acquire(f.bs[2]).Release()
		defer lease.Acquire(f.bs[2]).Release()
		full, _ := f.m.Reserve(f.bs, DefaultNodeBudgetTokens, nil)
		_ = f.m.WaitReady(context.Background(), full)
		j, err := f.m.Reserve(f.bs, 10, nil)
		if err != nil {
			t.Fatal(err)
		}
		if f.m.ModeOf(f.bs[1]) != Draining {
			t.Fatalf("pd=%v: modes %v, want node 1 draining", pd, f.m.ModeOf(f.bs[1]))
		}
		if j.Backend != f.bs[0] {
			t.Errorf("pd=%v: job placed on %s, want queued on the prefill node, not the draining one", pd, j.Backend.URL)
		}
		f.m.Tick(f.bs) // still over budget and still decoding: keeps draining
		if f.m.ModeOf(f.bs[1]) != Draining {
			t.Errorf("pd=%v: drain abandoned while the overflow lasts (mode %v)", pd, f.m.ModeOf(f.bs[1]))
		}
		decode.Release()
		f.m.Tick(f.bs)
		if f.m.ModeOf(f.bs[1]) != Prefill || j.Backend != f.bs[1] {
			t.Errorf("pd=%v: after the drain node 1 is %v and the job on %s; want prefill and the job moved there",
				pd, f.m.ModeOf(f.bs[1]), j.Backend.URL)
		}
		if err := f.m.WaitReady(context.Background(), j); err != nil {
			t.Fatal(err)
		}
		j.Done()
		full.Done()
	}
}

// A drain whose overflow clears before the node's decodes finish is abandoned,
// not completed.
func TestDrainIsAbandonedOnlyWhenTheOverflowClears(t *testing.T) {
	f := pdModeFleet(t, 3, Config{MaxFleetFraction: 0.7})
	defer lease.Acquire(f.bs[1]).Release() // node 1 keeps decoding
	defer lease.Acquire(f.bs[2]).Release()
	defer lease.Acquire(f.bs[2]).Release()
	full, _ := f.m.Reserve(f.bs, DefaultNodeBudgetTokens, nil)
	_ = f.m.WaitReady(context.Background(), full)
	q, _ := f.m.Reserve(f.bs, 10, nil) // drains node 1, queues on the floor
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[1]) != Draining {
		t.Fatalf("mode %v while the floor is over, want draining", f.m.ModeOf(f.bs[1]))
	}
	full.Done() // the overflow clears: q runs on the floor
	if err := f.m.WaitReady(context.Background(), q); err != nil {
		t.Fatal(err)
	}
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[1]) != Decode {
		t.Errorf("mode %v after the overflow cleared, want the drain abandoned (decode)", f.m.ModeOf(f.bs[1]))
	}
	q.Done()
}

// delayFleet runs the delay rule: drain when the p90 wait stays over 1s for
// 5s, release the newest converted node after 10s at or under 250ms.
func delayFleet(t *testing.T, n int) *fleet {
	return pdModeFleet(t, n, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.8, DrainDelay: time.Second,
		ConvertAfter: 5 * time.Second, ReleaseIdle: 10 * time.Second, CostPick: true})
}

func TestDelayRuleDrainsOnlyWhenTheWaitStaysHigh(t *testing.T) {
	f := delayFleet(t, 4)
	running, _ := f.m.Reserve(f.bs, 100, nil) // the floor's whole budget
	_ = f.m.WaitReady(context.Background(), running)
	waiting, _ := f.m.Reserve(f.bs, 50, nil) // over budget: waits in the router
	if f.m.State().Conversions != 0 {
		t.Fatal("a job over budget drained a node at once; the delay rule drains from Tick")
	}
	f.clk.Advance(2 * time.Second) // the waiting job is 2s old: p90 over 1s
	f.m.Tick(f.bs)
	f.clk.Advance(4 * time.Second)
	f.m.Tick(f.bs)
	if got := f.m.State().Conversions; got != 0 {
		t.Fatalf("drained after %v over the delay, want none before ConvertAfter (5s)", 4*time.Second)
	}
	f.clk.Advance(time.Second)
	f.m.Tick(f.bs)
	if got := f.m.State().Conversions; got != 1 {
		t.Fatalf("conversions = %d after 5s over the delay, want 1", got)
	}
	running.Done()
	waiting.Done()

	// A short wait that never crosses the delay drains nothing.
	g := delayFleet(t, 4)
	r, _ := g.m.Reserve(g.bs, 100, nil)
	_ = g.m.WaitReady(context.Background(), r)
	w, _ := g.m.Reserve(g.bs, 50, nil)
	g.clk.Advance(500 * time.Millisecond)
	g.m.Tick(g.bs)
	r.Done()
	_ = g.m.WaitReady(context.Background(), w)
	w.Done()
	g.clk.Advance(time.Minute)
	g.m.Tick(g.bs)
	if got := g.m.State().Conversions; got != 0 {
		t.Errorf("conversions = %d for a 500ms wait, want 0", got)
	}
}

// Low load: after ReleaseIdle under the low watermark the newest converted
// node goes back to decode, then the next newest after a window of its own;
// the minimum of one prefill node stays.
func TestLowDelayReleasesTheNewestConvertedNodeFirst(t *testing.T) {
	f := delayFleet(t, 5)
	// Two converted nodes, node 1 before node 2.
	f.m.mu.Lock()
	for i, b := range f.bs[1:3] {
		n := f.m.nodeLocked(b)
		n.mode, n.convertedAt = Prefill, f.clk.Now().Add(time.Duration(i+1)*time.Second)
	}
	f.m.mu.Unlock()
	// Node 1 has a leg running throughout, so it is never idle: only the
	// low-delay rule can release it.
	busy, _ := f.m.Reserve(f.bs[1:2], 10, nil)
	if err := f.m.WaitReady(context.Background(), busy); err != nil || busy.Backend != f.bs[1] {
		t.Fatalf("setup: busy leg on %s (%v), want node 1", busy.Backend.URL, err)
	}
	f.m.Tick(f.bs) // no waits: the signal is at its low watermark from now
	f.clk.Advance(9 * time.Second)
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[2]) != Prefill {
		t.Fatal("released before ReleaseIdle under the low watermark")
	}
	f.clk.Advance(time.Second)
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[2]) != Decode || f.m.ModeOf(f.bs[1]) != Prefill {
		t.Fatalf("modes %v, want the newest (node 2) released and node 1 kept", f.modes())
	}
	f.clk.Advance(10 * time.Second)
	f.m.Tick(f.bs)
	f.clk.Advance(time.Hour)
	f.m.Tick(f.bs)
	if n := count(f.modes(), Prefill); n != 1 {
		t.Errorf("modes %v, want only the minimum of one prefill node left", f.modes())
	}
}

// Legs pack: each goes to the first prefill node with room, oldest first
// (initial nodes, then by conversion time), so the newest is the last used.
func TestPDModeLegsPackOldestFirst(t *testing.T) {
	f := pdModeFleet(t, 4, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.8})
	f.m.mu.Lock()
	f.m.nodeLocked(f.bs[3]).mode, f.m.nodes[f.bs[3].URL].convertedAt = Prefill, f.clk.Now().Add(time.Minute)
	f.m.nodeLocked(f.bs[2]).mode, f.m.nodes[f.bs[2].URL].convertedAt = Prefill, f.clk.Now().Add(time.Second)
	f.m.mu.Unlock()
	var got []*registry.Backend
	for _, tok := range []int64{60, 60, 60, 10} {
		j, err := f.m.Reserve(f.bs, tok, func([]*registry.Backend, func(*registry.Backend) int64) (*registry.Backend, error) {
			t.Fatal("a policy was consulted for a packed leg")
			return nil, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		defer j.Done()
		got = append(got, j.Backend)
	}
	want := []*registry.Backend{f.bs[0], f.bs[2], f.bs[3], f.bs[0]}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("leg %d on %s, want %s (initial node 0, then node 2, then the newest node 3)", i, got[i].URL, want[i].URL)
		}
	}
}

func TestJobChargeAndWaitAreObserved(t *testing.T) {
	f := delayFleet(t, 3)
	j, _ := f.m.Reserve(f.bs, 70, nil)
	f.clk.Advance(300 * time.Millisecond)
	_ = f.m.WaitReady(context.Background(), j)
	j.Done()
	tokens := histogram(t, metrics.PrefillJobTokens.WithLabelValues(t.Name()))
	wait := histogram(t, metrics.PrefillJobWait.WithLabelValues(t.Name()))
	if tokens.GetSampleCount() != 1 || tokens.GetSampleSum() != 70 {
		t.Errorf("job tokens count %d sum %v, want 1 and 70", tokens.GetSampleCount(), tokens.GetSampleSum())
	}
	if wait.GetSampleCount() != 1 {
		t.Errorf("job wait count %d, want 1", wait.GetSampleCount())
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

func TestPDModeConversionAndReleaseAreLogged(t *testing.T) {
	var buf bytes.Buffer
	f := pdModeFleet(t, 3, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.7, Log: slog.New(slog.NewTextHandler(&buf, nil))})
	j1, _ := f.m.Reserve(f.bs, 100, nil)
	_ = f.m.WaitReady(context.Background(), j1)
	j2, _ := f.m.Reserve(f.bs, 10, nil) // drains an idle node and runs there
	_ = f.m.WaitReady(context.Background(), j2)
	j2.Done()
	j1.Done()
	out := buf.String()
	for _, want := range []string{
		`msg="decode node draining for prefill"`, "reason=no_prefill_room", "job_tokens=10", "node_budget_tokens=100",
		"prefill_pending_tokens=100", `msg="prefill node released to decode"`, "reason=idle",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("log lacks %q:\n%s", want, out)
		}
	}
	if strings.Count(out, "released to decode") != 1 {
		t.Errorf("want exactly one release (the floor stays):\n%s", out)
	}
}

// The initial prefill nodes are released like any other; only the minimum is
// kept.
func TestPDModeInitialNodesAreReleasedDownToTheMinimum(t *testing.T) {
	f := newFleet(t, 4, Config{})
	f.m = New(Config{PD: true, Initial: []string{f.bs[0].URL, f.bs[1].URL}, MinPrefill: 1,
		ReleaseIdle: time.Second, Clock: f.clk, PoolName: t.Name()})
	f.m.Tick(f.bs) // both initial nodes are prefill, idle from now
	if f.m.ModeOf(f.bs[0]) != Prefill || f.m.ModeOf(f.bs[1]) != Prefill {
		t.Fatalf("modes %v, want the two initial nodes prefill", f.modes())
	}
	f.clk.Advance(time.Second)
	f.m.Tick(f.bs)
	f.clk.Advance(time.Second)
	f.m.Tick(f.bs)
	if n := count(f.modes(), Prefill); n != 1 {
		t.Errorf("%d prefill nodes after both went idle, want the minimum 1 (modes %v)", n, f.modes())
	}
}

// With no initial nodes the minimum is reached by draining, and a drain toward
// the minimum is never abandoned for lack of load.
func TestPDModeMinimumIsReachedByDrainingAndKept(t *testing.T) {
	f := newFleet(t, 5, Config{})
	f.m = New(Config{PD: true, MinPrefill: 2, MaxFleetFraction: 0.6, Clock: f.clk, PoolName: t.Name()})
	busy := lease.Acquire(f.bs[0]) // decoding: a drain of it waits
	defer busy.Release()
	for _, b := range f.bs[1:] {
		defer lease.Acquire(b).Release()
		defer lease.Acquire(b).Release()
	}
	f.m.Tick(f.bs)
	f.m.Tick(f.bs)
	if n := count(f.modes(), Draining) + count(f.modes(), Prefill); n != 2 {
		t.Fatalf("%d nodes out of decode, want the minimum 2 (modes %v)", n, f.modes())
	}
	f.clk.Advance(time.Minute)
	f.m.Tick(f.bs) // nothing waits, yet the drains stand: they are the minimum
	if n := count(f.modes(), Decode); n != 3 {
		t.Errorf("modes %v: a drain toward the minimum was abandoned", f.modes())
	}
}
