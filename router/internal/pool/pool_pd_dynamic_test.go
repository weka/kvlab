package pool

import (
	"log/slog"
	"testing"

	"github.com/weka/kvlab/kvcache"
	"github.com/weka/kvlab/router/internal/clock"
	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/policy/affinity"
	"github.com/weka/kvlab/router/internal/prefill"
	"github.com/weka/kvlab/router/internal/registry"
)

// Under dynamic P/D a drained node keeps its decode marks while it drains (an
// abandoned drain returns with them) and loses them when it becomes a prefill
// node.
func TestDynamicPDForgetsDecodeMarksAtPromotionNotAtDrain(t *testing.T) {
	p, err := New(Config{
		Name: t.Name(), DefaultCapacity: 1, DefaultDialect: "openai",
		Backends:  []Backend{{URL: "http://d0:8000"}, {URL: "http://d1:8000"}, {URL: "http://d2:8000"}},
		Flow:      affinity.Config{CostRouting: true},
		PDDynamic: &prefill.Config{MaxFleetFraction: 0.7, NodeBudgetTokens: 1},
	}, clock.Real{}, slog.Default())
	if err != nil {
		t.Fatal(err)
	}
	bs := p.Registry.Snapshot().Backends
	for _, b := range bs {
		b.SetHealth(registry.Healthy)
	}
	rr := &policy.RoutingRequest{Units: []kvcache.Unit{{Hash: 1, Tokens: 256}, {Hash: 2, Tokens: 256}}, Model: "m"}
	mgr := p.PD.Dynamic
	// Make the node with the decode marks the drain target: the others have
	// older requests in flight.
	var target *registry.Backend
	for _, b := range bs[1:] {
		defer lease.Acquire(b).Release()
	}
	target = bs[0]
	p.Flow.Commit(target, rr)
	decoding := lease.Acquire(target)

	j, err := mgr.Reserve(bs, 10, nil) // no prefill node: drains one
	if err != nil {
		t.Fatal(err)
	}
	defer j.Done()
	if mgr.ModeOf(target) != prefill.Draining {
		t.Fatalf("setup: %s is %v, want draining", target.URL, mgr.ModeOf(target))
	}
	if p.Flow.HeldDepth(rr) == 0 {
		t.Fatal("decode marks dropped when the drain started")
	}
	decoding.Release()
	mgr.Tick(bs)
	if mgr.ModeOf(target) != prefill.Prefill {
		t.Fatalf("%s is %v after its decodes ended, want prefill", target.URL, mgr.ModeOf(target))
	}
	if p.Flow.HeldDepth(rr) != 0 {
		t.Error("decode marks kept after the node became a prefill node")
	}
}
