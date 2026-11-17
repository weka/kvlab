package prefill

import (
	"testing"
	"time"
)

// drainFleet: 5 nodes, up to 4 may be prefill, 100-token budget, drains at
// most every 5s, no convert-after hold.
func drainFleet(t *testing.T) *fleet {
	return newFleet(t, 5, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.8, DrainMinInterval: 5 * time.Second})
}

func (f *fleet) reserve(t *testing.T, tokens int64) *Job {
	t.Helper()
	j, err := f.m.Reserve(f.bs, tokens, nil)
	if err != nil {
		t.Fatal(err)
	}
	return j
}

func TestDrainMinIntervalAllowsOneDrainPerInterval(t *testing.T) {
	f := drainFleet(t)
	first := f.reserve(t, 100) // no prefill node yet: drains at once
	defer first.Done()
	if got := f.m.State().Conversions; got != 1 {
		t.Fatalf("conversions = %d, want the first drain", got)
	}
	// Over budget 1s later: a drain is due but held back; the job queues.
	f.clk.Advance(time.Second)
	held := f.reserve(t, 100)
	defer held.Done()
	if held.Backend != first.Backend {
		t.Errorf("held job on %s, want queued on the existing prefill node", held.Backend.URL)
	}
	st := f.m.State()
	if st.Conversions != 1 || st.DrainsDeferred != 1 {
		t.Fatalf("conversions = %d, deferred = %d inside the interval; want 1 and 1", st.Conversions, st.DrainsDeferred)
	}
	// Still over budget once the interval has passed: the second drain.
	f.clk.Advance(4 * time.Second)
	second := f.reserve(t, 10)
	defer second.Done()
	if got := f.m.State().Conversions; got != 2 {
		t.Errorf("conversions = %d after the interval, want 2", got)
	}
	if second.Backend == first.Backend || f.m.ModeOf(second.Backend) != Draining {
		t.Errorf("job after the interval on %s (%v), want a newly drained node", second.Backend.URL, f.m.ModeOf(second.Backend))
	}
}

func TestDrainMinIntervalLetsAnOverflowClearWithoutASecondDrain(t *testing.T) {
	f := drainFleet(t)
	a := f.reserve(t, 60)
	defer a.Done()
	b := f.reserve(t, 60) // 120 > 100: a drain is due, held back
	f.clk.Advance(2 * time.Second)
	b.Done() // the burst finishes inside the interval
	c := f.reserve(t, 30)
	defer c.Done()
	f.clk.Advance(10 * time.Second)
	d := f.reserve(t, 10)
	defer d.Done()
	st := f.m.State()
	if st.Conversions != 1 {
		t.Errorf("conversions = %d, want only the first: the overflow cleared before the interval ended", st.Conversions)
	}
	if st.DrainsDeferred != 1 {
		t.Errorf("deferred = %d, want 1", st.DrainsDeferred)
	}
}

func TestDrainMinIntervalZeroIsNoLimit(t *testing.T) {
	f := newFleet(t, 5, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.8})
	for range 3 {
		defer f.reserve(t, 100).Done()
	}
	if got := f.m.State().Conversions; got != 3 {
		t.Errorf("conversions = %d, want 3 with no interval", got)
	}
}
