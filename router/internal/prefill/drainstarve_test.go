package prefill

import (
	"context"
	"errors"
	"testing"
	"time"
)

// skipFleet: nodes 0 prefill, the rest decode; the starvation and
// underutilization rules on (starveFraction, idleFraction) and drains skipping
// starved nodes when skip.
func skipFleet(t *testing.T, n int, skip bool, starveFraction, idleFraction float64, minPrefill int, readings map[string]probeReading) *fleet {
	t.Helper()
	return pdModeFleet(t, n, Config{
		MaxFleetFraction: 0.8, MinPrefill: minPrefill,
		TTFTWindow: 4, TTFTHigh: 5 * time.Second, TTFTLow: time.Second,
		StarveKV: 0.95, StarveWaiting: 1, StarveFraction: starveFraction,
		StarvedShrinkGuard: 5 * time.Second, StarvedTTFTHigh: 10 * time.Second,
		IdleKV: 0.6, IdleFraction: idleFraction, UnderutilTTFTHigh: 2 * time.Second,
		DrainSkipStarved: skip,
		DecodeProbe:      probeOf(readings),
	})
}

var (
	rStarved = probeReading{kv: 0.99, waiting: 2}
	rBusy    = probeReading{kv: 0.8}
	rIdle    = probeReading{kv: 0.1}
)

func (f *fleet) drained() []int {
	var out []int
	for i, b := range f.bs {
		if f.m.nodes[b.URL].mode == Draining {
			out = append(out, i)
		}
	}
	return out
}

// Every drain reason picks among the decode nodes not starved by their latest
// sample. Node 0 prefill (and node 1 unless min_nodes); the decode readings
// are for nodes 2-4.
func TestDrainSkipsStarvedNodesForEveryReason(t *testing.T) {
	for _, c := range []struct {
		name                         string
		starveFraction, idleFraction float64
		minPrefill                   int
		decode                       [3]probeReading
		ttft                         time.Duration
		reason                       string
		allowed                      map[int]bool
	}{
		{"ttft_high", 0.9, 0, 1, [3]probeReading{rStarved, rStarved, rBusy}, 5 * time.Second, "ttft_high", map[int]bool{4: true}},
		{"decode_starved_ttft_high", 0.5, 0, 1, [3]probeReading{rStarved, rStarved, rBusy}, 10 * time.Second, "decode_starved_ttft_high", map[int]bool{4: true}},
		{"underutil_ttft_high", 0.5, 0.5, 1, [3]probeReading{rStarved, rIdle, rIdle}, 3 * time.Second, "underutil_ttft_high", map[int]bool{3: true, 4: true}},
		{"min_nodes", 0.9, 0, 3, [3]probeReading{rStarved, rStarved, rBusy}, 0, "min_nodes", map[int]bool{1: true, 4: true}},
	} {
		t.Run(c.name, func(t *testing.T) {
			readings := map[string]probeReading{}
			f := skipFleet(t, 5, true, c.starveFraction, c.idleFraction, c.minPrefill, readings)
			if c.reason != "min_nodes" {
				f.setMode(f.bs[1], Prefill, f.clk.Now().Add(time.Second))
			} else {
				readings[f.bs[1].URL] = rStarved
				c.allowed = map[int]bool{4: true} // node 1 is decode and starved too
			}
			for i, r := range c.decode {
				readings[f.bs[2+i].URL] = r
			}
			f.m.SampleDecode(context.Background(), f.bs)
			if c.ttft > 0 {
				f.ttft(4, c.ttft)
			}
			f.m.Tick(f.bs)
			d := f.drained()
			if len(d) != 1 || !c.allowed[d[0]] {
				t.Fatalf("drained %v, want one of %v (modes %v)", d, c.allowed, f.modes())
			}
			if got := actionCount(t, "drain", c.reason); got != 1 {
				t.Errorf("drain{reason=%s} = %v, want 1", c.reason, got)
			}
			if got := gauge(t, f.m.cfg.PDMetrics.DrainCandidates); got != float64(len(c.allowed)) {
				t.Errorf("drain candidates %v, want %d", got, len(c.allowed))
			}
		})
	}
}

// With every decode node starved nothing is drained: one hold that restarts
// the window, not one per tick.
func TestDrainBlockedWhenAllDecodeStarved(t *testing.T) {
	readings := map[string]probeReading{}
	f := skipFleet(t, 5, true, 0.5, 0, 1, readings)
	f.setMode(f.bs[1], Prefill, f.clk.Now().Add(time.Second))
	for _, b := range f.bs {
		readings[b.URL] = rStarved
	}
	f.m.SampleDecode(context.Background(), f.bs)
	f.ttft(4, 10*time.Second) // would be decode_starved_ttft_high
	f.m.Tick(f.bs)
	if d := f.drained(); len(d) != 0 {
		t.Fatalf("drained %v with every decode node starved", d)
	}
	if got := actionCount(t, "hold", "drain_blocked_all_starved"); got != 1 {
		t.Fatalf("hold{drain_blocked_all_starved} = %v, want 1", got)
	}
	f.m.mu.Lock()
	restarted := f.m.lastAction.Equal(f.clk.Now()) && f.m.ttft.since == 0 && len(f.m.starve) == 0
	f.m.mu.Unlock()
	if !restarted {
		t.Error("the blocked hold did not restart the window")
	}
	if got := gauge(t, f.m.cfg.PDMetrics.DrainCandidates); got != 0 {
		t.Errorf("drain candidates %v, want 0", got)
	}
	f.m.Tick(f.bs)
	if got := actionCount(t, "hold", "drain_blocked_all_starved"); got != 1 {
		t.Errorf("held %v times on one window, want 1", got)
	}
}

// Below the minimum with every decode node starved: paced holds, not one per
// tick.
func TestDrainBlockedMinNodesIsPaced(t *testing.T) {
	readings := map[string]probeReading{}
	f := skipFleet(t, 4, true, 0.9, 0, 2, readings)
	for _, b := range f.bs {
		readings[b.URL] = rStarved
	}
	f.m.SampleDecode(context.Background(), f.bs)
	f.m.Tick(f.bs)
	f.m.Tick(f.bs)
	f.clk.Advance(time.Second)
	f.m.Tick(f.bs)
	if got := actionCount(t, "hold", "drain_blocked_all_starved"); got != 1 || len(f.drained()) != 0 {
		t.Fatalf("holds %v drained %v, want one hold, no drain", got, f.drained())
	}
	f.clk.Advance(5 * time.Second) // DrainMinInterval
	f.m.Tick(f.bs)
	if got := actionCount(t, "hold", "drain_blocked_all_starved"); got != 2 {
		t.Errorf("holds %v after the interval, want 2", got)
	}
}

// A node with no sample yet, or an unread one, is not starved.
func TestDrainNoSampleNodesAreCandidates(t *testing.T) {
	readings := map[string]probeReading{}
	f := skipFleet(t, 4, true, 0.9, 0, 2, readings)
	f.m.Tick(f.bs) // never sampled
	if d := f.drained(); len(d) != 1 {
		t.Fatalf("drained %v with no sample, want one", d)
	}
	readings = map[string]probeReading{}
	f = skipFleet(t, 4, true, 0.9, 0, 2, readings)
	readings[f.bs[1].URL] = rStarved
	readings[f.bs[2].URL] = rStarved
	readings[f.bs[3].URL] = probeReading{err: errors.New("down")}
	f.m.SampleDecode(context.Background(), f.bs)
	f.m.Tick(f.bs)
	if d := f.drained(); len(d) != 1 || d[0] != 3 {
		t.Errorf("drained %v, want the unread node 3", d)
	}
}

// --pd-drain-skip-starved=false: the old choice among all decode nodes.
func TestDrainSkipStarvedOffKeepsOldBehaviour(t *testing.T) {
	readings := map[string]probeReading{}
	f := skipFleet(t, 5, false, 0.5, 0, 1, readings)
	f.setMode(f.bs[1], Prefill, f.clk.Now().Add(time.Second))
	for _, b := range f.bs {
		readings[b.URL] = rStarved
	}
	f.m.SampleDecode(context.Background(), f.bs)
	f.ttft(4, 10*time.Second)
	f.m.Tick(f.bs)
	if d := f.drained(); len(d) != 1 {
		t.Fatalf("drained %v, want one starved node as before", d)
	}
	if got := actionCount(t, "drain", "decode_starved_ttft_high"); got != 1 {
		t.Errorf("drain{decode_starved_ttft_high} = %v, want 1", got)
	}
	if got := actionCount(t, "hold", "drain_blocked_all_starved"); got != 0 {
		t.Errorf("blocked hold %v with the flag off", got)
	}
	if got := gauge(t, f.m.cfg.PDMetrics.DrainCandidates); got != 3 {
		t.Errorf("drain candidates %v, want all 3 decode nodes", got)
	}
}
