package prefill

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/registry"
)

// nodeState is one backend's row of State, zero for a node the manager does not
// track.
func (f *fleet) nodeState(b *registry.Backend) NodeState {
	for _, n := range f.m.State().Nodes {
		if n.URL == b.URL {
			return n
		}
	}
	return NodeState{}
}

// assertDrained checks that no node holds pending tokens, running jobs or
// queued jobs.
func (f *fleet) assertDrained(t *testing.T) {
	t.Helper()
	for _, n := range f.m.State().Nodes {
		if n.PendingTokens != 0 || n.RunningJobs != 0 || n.QueuedJobs != 0 {
			t.Errorf("%s after every job finished: pending %d, running %d, queued %d",
				n.URL, n.PendingTokens, n.RunningJobs, n.QueuedJobs)
		}
	}
}

// waitAll runs WaitReady for every job in the background and reports on ready
// the jobs as they get their slot.
func (f *fleet) waitAll(ctx context.Context, jobs []*Job) <-chan *Job {
	ready := make(chan *Job, len(jobs))
	for _, j := range jobs {
		go func() {
			if f.m.WaitReady(ctx, j) == nil {
				ready <- j
			}
		}()
	}
	return ready
}

// A cold start rejects a burst of first turns at once. With ConvertAfter set
// there is one prefill node for its first seconds, and every job of the burst
// queues on it. The node must still be given no more than its budget at once:
// the rest wait in the manager, and move to the next node that converts, rather
// than all being sent to the first node to run there one after another for
// minutes while the nodes converted after it sit idle.
func TestColdBurstRunsNoMoreThanTheBudgetOnOneNodeAndSpreadsWhenNodesConvert(t *testing.T) {
	const budget, job, burst = 250_000, 55_000, 60
	f := newFleet(t, 8, Config{NodeBudgetTokens: budget, ConvertAfter: 10 * time.Second,
		MaxFleetFraction: 0.5, NodeConcurrency: 256})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var jobs []*Job
	for range burst {
		j, err := f.m.Reserve(f.bs, job, nil)
		if err != nil {
			t.Fatal(err)
		}
		jobs = append(jobs, j)
	}
	first := jobs[0].Backend
	ready := f.waitAll(ctx, jobs)
	perRun := int(budget / job) // 4 jobs of 55k fit a 250k budget
	waitFor(t, func() bool { return f.nodeState(first).RunningJobs >= perRun })
	time.Sleep(20 * time.Millisecond) // let any over-granting show
	if n := f.nodeState(first).RunningJobs; n > perRun {
		t.Fatalf("first prefill node runs %d jobs of %d tokens at once, %d tokens against a %d budget",
			n, job, int64(n)*job, budget)
	}

	// Ten seconds over budget: the next job converts a second node, and the
	// burst's waiting jobs move onto it up to its budget.
	f.clk.Advance(10 * time.Second)
	late, err := f.m.Reserve(f.bs, job, nil)
	if err != nil {
		t.Fatal(err)
	}
	if late.Backend == first {
		t.Fatal("no second node converted after 10s over budget")
	}
	second := late.Backend
	if err := f.m.WaitReady(ctx, late); err != nil {
		t.Fatal(err)
	}
	waitFor(t, func() bool { return f.nodeState(second).RunningJobs >= perRun })
	// The new node runs late plus three of the burst's waiting jobs.
	if p := f.nodeState(first).PendingTokens; p > int64(burst-(perRun-1))*job {
		t.Errorf("first node still holds %d pending tokens: none of its waiting jobs moved to the new node", p)
	}

	// Finish every job as it gets its slot.
	late.Done()
	for done := 0; done < burst; done++ {
		select {
		case j := <-ready:
			j.Done()
		case <-time.After(5 * time.Second):
			t.Fatalf("%d of %d jobs finished; the rest never got a slot (%+v)", done, burst, f.m.State().Nodes)
		}
	}
	f.assertDrained(t)
}

// Every way a job can end returns its tokens: success, failure, a job held
// through 429 retries, a caller that leaves while queued behind the
// concurrency limit, and one that leaves while its node is still draining,
// with nodes released and converted again between cycles. After each cycle
// every node is back to zero and in decode.
func TestPendingReturnsToZeroOnEveryJobPath(t *testing.T) {
	f := newFleet(t, 6, Config{NodeBudgetTokens: 100, NodeConcurrency: 2, MaxFleetFraction: 0.5})
	bg := context.Background()
	reserve := func(tokens int64) *Job {
		t.Helper()
		j, err := f.m.Reserve(f.bs, tokens, nil)
		if err != nil {
			t.Fatal(err)
		}
		return j
	}
	ready := func(j *Job) *Job {
		t.Helper()
		if err := f.m.WaitReady(bg, j); err != nil {
			t.Fatal(err)
		}
		return j
	}

	for cycle := range 3 {
		ok, failed := ready(reserve(40)), ready(reserve(30))

		// Fits the budget but not the concurrency limit of 2: it queues, and
		// its caller leaves.
		queued := reserve(20)
		if queued.Backend != ok.Backend {
			t.Fatalf("cycle %d: setup: the third job is not on the first node", cycle)
		}
		ctx, cancel := context.WithCancel(bg)
		left := make(chan error, 1)
		go func() { left <- f.m.WaitReady(ctx, queued) }()
		waitFor(t, func() bool { return f.nodeState(ok.Backend).QueuedJobs == 1 })
		cancel()
		if err := <-left; err == nil {
			t.Fatalf("cycle %d: a queued job was granted past the concurrency limit", cycle)
		}
		queued.Done()

		// Over the first node's budget: a second node converts. The job holds
		// its slot as one being retried after 429s would.
		retried := ready(reserve(90))
		if retried.Backend == ok.Backend {
			t.Fatalf("cycle %d: setup: the over-budget job did not convert a second node", cycle)
		}
		ok.Done()
		failed.Done()
		retried.Done()

		// Every node is busy with decode, so a conversion drains, and the
		// caller leaves before it finishes.
		var leases []*lease.Lease
		for _, b := range f.bs {
			leases = append(leases, lease.Acquire(b))
		}
		draining := reserve(10)
		dctx, dcancel := context.WithTimeout(bg, 10*time.Millisecond)
		if err := f.m.WaitReady(dctx, draining); err == nil {
			t.Fatalf("cycle %d: WaitReady returned before the draining node's decode finished", cycle)
		}
		dcancel()
		draining.Done()
		for _, l := range leases {
			l.Release()
		}

		f.assertDrained(t)
		if got := count(f.modes(), Decode); got != len(f.bs) {
			t.Errorf("cycle %d: %d of %d nodes back in decode", cycle, got, len(f.bs))
		}
	}
	if r := f.m.State().Releases; r < 9 {
		t.Errorf("releases = %d, want 3 per cycle", r)
	}
}

// A node can leave the pool, by discovery, while one of its jobs runs, and come
// back before that job ends. The job ending must settle only the node it ran
// on: the returning node's mode, pending tokens and pending metric are its own,
// and the departed node going idle is not a mode change of the backend.
func TestJobEndingAfterItsNodeLeftLeavesTheReturnedNodeAlone(t *testing.T) {
	var toDecode []string
	f := newFleet(t, 3, Config{OnModeChange: func(b *registry.Backend, from, to Mode) {
		if to == Decode {
			toDecode = append(toDecode, b.URL)
		}
	}})
	bg := context.Background()
	old, _ := f.m.Reserve(f.bs, 70, nil)
	if err := f.m.WaitReady(bg, old); err != nil {
		t.Fatal(err)
	}
	b := old.Backend
	var rest []*registry.Backend
	for _, x := range f.bs {
		if x != b {
			rest = append(rest, x)
		}
	}
	f.m.Prune(rest)

	// It returns, and is converted afresh for a new job: the only decode node
	// without load.
	for _, x := range rest {
		defer lease.Acquire(x).Release()
	}
	fresh, err := f.m.Reserve(f.bs, 25, nil)
	if err != nil {
		t.Fatal(err)
	}
	if fresh.Backend != b {
		t.Fatalf("setup: new job on %s, want the returned node %s", fresh.Backend.URL, b.URL)
	}
	if err := f.m.WaitReady(bg, fresh); err != nil {
		t.Fatal(err)
	}

	toDecode = nil
	old.Done()
	if len(toDecode) != 0 {
		t.Errorf("the departed node's last job ending moved %v to decode", toDecode)
	}
	if n := f.nodeState(b); n.Mode != Prefill.String() || n.PendingTokens != 25 {
		t.Errorf("returned node after the old job ended: %+v, want prefill with 25 pending", n)
	}
	if g := testutil.ToFloat64(metrics.PrefillPendingTokens.WithLabelValues(t.Name(), b.URL)); g != 25 {
		t.Errorf("pending metric for the returned node = %v, want 25", g)
	}
	fresh.Done()
	f.assertDrained(t)
}
