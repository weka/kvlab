package prefill

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/lease"
)

// snowballFleet: n decode nodes all busy decoding, so a drained node stays
// draining; budget 100 tokens, drains at once, at most one per 5s.
func snowballFleet(t *testing.T, n, maxDraining int) *fleet {
	t.Helper()
	f := newFleet(t, n, Config{MaxFleetFraction: 0.9, NodeBudgetTokens: 100,
		DrainMinInterval: 5 * time.Second, MaxDraining: maxDraining})
	for _, b := range f.bs {
		l := lease.Acquire(b)
		t.Cleanup(l.Release)
	}
	return f
}

func (f *fleet) draining() int { return count(f.modes(), Draining) }

// A sustained overflow (jobs arriving faster than anything finishes) drains
// at most MaxDraining nodes at once, however long it lasts.
func TestRejectDrainsCappedUnderSustainedOverflow(t *testing.T) {
	for _, c := range []struct {
		max, want int
	}{{2, 2}, {1, 1}} {
		f := snowballFleet(t, 12, c.max)
		peak := 0
		for range 24 { // two minutes of overflow, a job every 5s
			if _, err := f.m.Reserve(f.bs, 100, nil); err != nil {
				t.Fatal(err)
			}
			peak = max(peak, f.draining())
			f.clk.Advance(5 * time.Second)
		}
		if peak != c.want || f.m.conversions.Load() != uint64(c.want) {
			t.Errorf("max %d: peak draining %d, conversions %d; want %d", c.max, peak, f.m.conversions.Load(), c.want)
		}
	}
}

// With no cap (0, the old behaviour) the same overflow keeps draining.
func TestRejectDrainsUncappedKeepDraining(t *testing.T) {
	f := snowballFleet(t, 12, 0)
	for range 24 {
		if _, err := f.m.Reserve(f.bs, 100, nil); err != nil {
			t.Fatal(err)
		}
		f.clk.Advance(5 * time.Second)
	}
	if got := f.draining(); got <= 2 {
		t.Errorf("draining %d, want the uncapped overflow to keep draining", got)
	}
}

// A node already draining is a budget of room on its way: no further drain
// while the pending tokens fit the prefill plus draining budgets.
func TestRejectNoDrainWhileDrainingCapacityCoversTheOverflow(t *testing.T) {
	f := snowballFleet(t, 6, 0)
	f.setMode(f.bs[0], Prefill, f.clk.Now())
	f.setMode(f.bs[1], Draining, f.clk.Now())
	if _, err := f.m.Reserve(f.bs, 100, nil); err != nil { // fills the prefill node
		t.Fatal(err)
	}
	f.clk.Advance(10 * time.Second)
	j, err := f.m.Reserve(f.bs, 80, nil) // 180 pending <= 2 x 100
	if err != nil {
		t.Fatal(err)
	}
	if f.draining() != 1 || j.Backend != f.bs[0] {
		t.Fatalf("drained for an overflow the draining node covers: draining %d, job on %s", f.draining(), j.Backend.URL)
	}
	// The overflow holding steady (one job in, one out) never drains again.
	for range 10 {
		f.clk.Advance(5 * time.Second)
		j2, err := f.m.Reserve(f.bs, 10, nil)
		if err != nil {
			t.Fatal(err)
		}
		j2.Done()
	}
	if f.draining() != 1 {
		t.Fatalf("steady overflow drained: draining %d", f.draining())
	}
	f.clk.Advance(5 * time.Second)
	if _, err := f.m.Reserve(f.bs, 50, nil); err != nil { // 230 > 200
		t.Fatal(err)
	}
	if f.draining() != 2 {
		t.Errorf("draining %d, want a second drain once the overflow outgrows the draining room", f.draining())
	}
}

// Reject mode logs its drains and releases as pd mode does.
func TestRejectModeLogsDrainsAndReleases(t *testing.T) {
	var buf bytes.Buffer
	f := newFleet(t, 4, Config{MaxFleetFraction: 0.9, NodeBudgetTokens: 100,
		Log: slog.New(slog.NewTextHandler(&buf, nil))})
	j, err := f.m.Reserve(f.bs, 100, nil)
	if err != nil {
		t.Fatal(err)
	}
	j.Done()
	f.m.mu.Lock()
	f.m.releaseForLocked(f.m.nodes[j.Backend.URL], "idle")
	f.m.mu.Unlock()
	out := buf.String()
	for _, want := range []string{"decode node draining for prefill", "reason=no_prefill_room", "prefill node released to decode"} {
		if !strings.Contains(out, want) {
			t.Errorf("log lacks %q:\n%s", want, out)
		}
	}
}
