package prefill

import (
	"context"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/lease"
)

// waitReadyNow runs WaitReady and reports whether it returned within a short
// real-time window, advancing the fake clock so its drain polls fire.
func (f *fleet) waitReadyNow(t *testing.T, j *Job) bool {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- f.m.WaitReady(ctx, j) }()
	deadline := time.After(100 * time.Millisecond)
	for {
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("WaitReady: %v", err)
			}
			return true
		case <-deadline:
			cancel()
			<-done
			return false
		default:
			f.clk.Advance(drainPoll)
			time.Sleep(time.Millisecond)
		}
	}
}

// overBudgetFleet has one prefill node running 80 of a 100-token budget and no
// room to convert another (the fleet cap is one node), then reserves a
// 50-token job, which can only go to that node.
func overBudgetFleet(t *testing.T, soft bool) (f *fleet, first, over *Job) {
	f = newFleet(t, 4, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.25, BudgetSoft: soft})
	first = f.reserve(t, 80)
	if !f.waitReadyNow(t, first) {
		t.Fatal("first job never started on the idle drained node")
	}
	over = f.reserve(t, 50)
	if over.Backend != first.Backend {
		t.Fatalf("over-budget job on %s, want the one prefill node %s", over.Backend.URL, first.Backend.URL)
	}
	return f, first, over
}

func TestSoftBudgetSendsAnOverBudgetJobAtOnce(t *testing.T) {
	f, first, over := overBudgetFleet(t, true)
	if !f.waitReadyNow(t, over) {
		t.Fatal("soft budget: over-budget job waited in the router")
	}
	st := f.nodeState(first.Backend)
	if st.RunningTokens != 130 || st.RunningJobs != 2 || st.QueuedJobs != 0 || st.PendingTokens != 130 {
		t.Errorf("state = %+v, want 2 jobs running 130 tokens over the 100 budget, none queued", st)
	}
	over.Done()
	first.Done()
	st = f.nodeState(first.Backend)
	if st.RunningTokens != 0 || st.PendingTokens != 0 || st.RunningJobs != 0 {
		t.Errorf("state after both finished = %+v, want all zero", st)
	}
}

func TestHardBudgetQueuesAnOverBudgetJob(t *testing.T) {
	f, first, over := overBudgetFleet(t, false)
	st := f.nodeState(first.Backend)
	if st.RunningTokens != 80 || st.QueuedJobs != 1 || st.PendingTokens != 130 {
		t.Fatalf("state = %+v, want 80 running and the 50-token job queued", st)
	}
	if f.waitReadyNow(t, over) {
		t.Fatal("hard budget: over-budget job started next to 80 running tokens")
	}
	// The cancelled wait dequeued it; reserve it again and free the room.
	over.Done()
	again := f.reserve(t, 50)
	first.Done()
	if !f.waitReadyNow(t, again) {
		t.Fatal("job never started once the budget had room")
	}
	again.Done()
}

func TestSoftBudgetStillConvertsAndADrainingNodeStillWaits(t *testing.T) {
	f := newFleet(t, 4, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.5, BudgetSoft: true})
	// Every node is decoding: node 0 least, so it is drained first, then
	// node 1.
	busy0 := lease.Acquire(f.bs[0])
	for i, b := range f.bs[1:] {
		for range i + 2 {
			defer lease.Acquire(b).Release()
		}
	}

	first := f.reserve(t, 80)
	if first.Backend != f.bs[0] || f.m.ModeOf(f.bs[0]) != Draining {
		t.Fatalf("first job on %s (%v), want node 0 draining", first.Backend.URL, f.m.ModeOf(first.Backend))
	}
	if f.waitReadyNow(t, first) {
		t.Fatal("soft budget: job started on a node still running decode work")
	}
	busy0.Release()
	if !f.waitReadyNow(t, first) {
		t.Fatal("job never started after its node drained")
	}

	// 80+50 is over budget: node 1 drains, and the job, which would wait for
	// node 1's decodes there, runs over budget on node 0 instead.
	over := f.reserve(t, 50)
	if got := f.m.State().Conversions; got != 2 {
		t.Fatalf("conversions = %d, want the over-budget job to drain a second node", got)
	}
	if f.m.ModeOf(f.bs[1]) != Draining {
		t.Fatalf("node 1 mode = %v, want draining", f.m.ModeOf(f.bs[1]))
	}
	if over.Backend != f.bs[0] {
		t.Fatalf("over-budget job on %s, want the prefill node", over.Backend.URL)
	}
	if !f.waitReadyNow(t, over) {
		t.Fatal("soft budget: over-budget job waited in the router")
	}
	if st := f.nodeState(f.bs[0]); st.RunningTokens != 130 || st.QueuedJobs != 0 {
		t.Errorf("node 0 = %+v, want 130 running tokens, none queued", st)
	}
	over.Done()
	first.Done()
}
