package pool_test

import (
	"context"
	"log/slog"
	"testing"

	"github.com/weka/kvlab/kvcache"
	"github.com/weka/kvlab/router/internal/clock"
	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/policy/affinity"
	"github.com/weka/kvlab/router/internal/pool"
	"github.com/weka/kvlab/router/internal/prefill"
	"github.com/weka/kvlab/router/internal/registry"
)

func newSplitPool(t *testing.T, cost bool) (*pool.Pool, []*registry.Backend) {
	t.Helper()
	p, err := pool.New(pool.Config{
		Name:            "default",
		Backends:        []pool.Backend{{URL: "http://a:8000"}, {URL: "http://b:8000"}, {URL: "http://c:8000"}},
		Flow:            affinity.Config{CostRouting: cost},
		PrefillSplit:    &prefill.Config{ReleaseIdle: 0},
		DefaultCapacity: 1,
	}, clock.Real{}, slog.Default())
	if err != nil {
		t.Fatal(err)
	}
	bs := p.Registry.Snapshot().Backends
	for _, b := range bs {
		b.SetHealth(registry.Healthy)
	}
	return p, bs
}

var splitReq = func() *policy.RoutingRequest {
	return &policy.RoutingRequest{Model: "m", Units: []kvcache.Unit{{Hash: 1, Tokens: 100}, {Hash: 2, Tokens: 100}}}
}

// uncached is the tokens of the request a flow would have b prefill: what its
// own tree does not hold for b.
func uncached(t *testing.T, f *affinity.Policy, b *registry.Backend) int64 {
	t.Helper()
	rr := splitReq()
	if _, err := f.Select(context.Background(), []*registry.Backend{b}, rr); err != nil {
		t.Fatal(err)
	}
	return rr.Prefill.Tokens
}

// The prefill flow exists only under cost routing, and is a flow of its own:
// a prefix committed to it is unknown to the decode flow, and the reverse.
func TestSplitFlowIsSeparateAndOnlyUnderCostRouting(t *testing.T) {
	if p, _ := newSplitPool(t, false); p.SplitFlow != nil {
		t.Fatal("the ladder got a prefill flow")
	}
	p, bs := newSplitPool(t, true)
	if p.SplitFlow == nil || p.SplitFlow == p.Flow {
		t.Fatal("cost routing with the split needs a prefill flow of its own")
	}
	a := bs[0]
	p.SplitFlow.Commit(a, splitReq())
	if got := uncached(t, p.SplitFlow, a); got != 0 {
		t.Errorf("prefill flow after its own commit: uncached %d, want 0", got)
	}
	if got := uncached(t, p.Flow, a); got != 200 {
		t.Errorf("decode flow saw the prefill flow's mark: uncached %d, want 200", got)
	}
	p.Flow.Commit(bs[1], splitReq())
	if got := uncached(t, p.SplitFlow, bs[1]); got != 200 {
		t.Errorf("prefill flow saw the decode flow's mark: uncached %d, want 200", got)
	}
}

// A node's marks count only while it is in the role that made them: leaving
// decode forgets its decode marks, leaving prefill forgets its prefill marks,
// and neither tree touches the other's nodes.
func TestRoleConversionAndReleaseForgetStaleMarks(t *testing.T) {
	p, bs := newSplitPool(t, true)
	a, b, c := bs[0], bs[1], bs[2]
	p.Flow.Commit(a, splitReq()) // a and b hold the prefix as decode nodes
	p.Flow.Commit(b, splitReq())
	// Load b and c so a is the least loaded and the one that gets drained.
	lease.Acquire(b)
	lease.Acquire(c)

	job, err := p.Split.Reserve(bs, 100, nil)
	if err != nil {
		t.Fatal(err)
	}
	if job.Backend != a {
		t.Fatalf("job on %s, want the least-loaded %s", job.Backend.URL, a.URL)
	}
	if got := uncached(t, p.Flow, a); got != 200 {
		t.Errorf("a left decode but the decode flow still credits it: uncached %d, want 200", got)
	}
	if got := uncached(t, p.Flow, b); got != 0 {
		t.Errorf("b stayed decode and lost its marks: uncached %d, want 0", got)
	}

	if err := p.Split.WaitReady(context.Background(), job); err != nil {
		t.Fatal(err)
	}
	p.SplitFlow.Commit(a, splitReq()) // the job succeeded
	if got := uncached(t, p.SplitFlow, a); got != 0 {
		t.Errorf("prefill flow after the job: uncached %d, want 0", got)
	}
	if got := uncached(t, p.Flow, a); got != 200 {
		t.Errorf("the job's prefix leaked into the decode flow: uncached %d, want 200", got)
	}

	job.Done() // ReleaseIdle 0: a is released to decode at once
	if m := p.Split.ModeOf(a); m != prefill.Decode {
		t.Fatalf("a is %v, want decode after release", m)
	}
	if got := uncached(t, p.SplitFlow, a); got != 200 {
		t.Errorf("a left prefill but the prefill flow still credits it: uncached %d, want 200", got)
	}
}
