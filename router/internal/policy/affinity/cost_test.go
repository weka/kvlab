package affinity

import (
	"context"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/lease"
	"github.com/weka/wekai/router/internal/registry"
)

func newCostPolicy(t *testing.T) *Policy {
	t.Helper()
	p, err := New(Config{CostRouting: true, TailTTL: testTTL, Clock: clock.NewFake(time.Time{})}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// A request of 768 tokens held on w0. Moving it costs its full 768 tokens, so the
// holder keeps it until its extra load exceeds that.
func TestCostRoutingStaysWhileExtraLoadIsBelowMoveCost(t *testing.T) {
	p := newCostPolicy(t)
	f := fleet(t, 2)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))

	f[0].AddActiveTokens(767)
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[0] {
		t.Errorf("routed to %s, want the holder while extra load 767 < move cost 768", got.URL)
	}
}

func TestCostRoutingMovesOnceExtraLoadExceedsMoveCost(t *testing.T) {
	p := newCostPolicy(t)
	f := fleet(t, 2)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))

	f[0].AddActiveTokens(769)
	before := decisions(t, "cost_move")
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[1] {
		t.Errorf("routed to %s, want the other backend once extra load 769 > move cost 768", got.URL)
	}
	if d := decisions(t, "cost_move") - before; d != 1 {
		t.Errorf("cost_move delta = %v, want 1", d)
	}
}

// Credit is only for the part of the prefix a backend holds, so a backend holding
// the shared start of a prompt beats one holding nothing at equal load.
func TestCostRoutingCreditsPartialPrefix(t *testing.T) {
	p := newCostPolicy(t)
	f := fleet(t, 3)
	for _, b := range f {
		p.AddBackend(b)
	}
	p.Commit(f[1], req(units(1, 2)))
	got, err := p.Select(context.Background(), f, req(units(1, 2, 3, 4)))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[1] {
		t.Errorf("routed to %s, want the backend holding half the prefix", got.URL)
	}
}

// A signal still removes a backend before cost is compared.
func TestCostRoutingOnlyConsidersUsableBackends(t *testing.T) {
	p, err := New(Config{CostRouting: true, NodeConcurrency: 1, TailTTL: testTTL, Clock: clock.NewFake(time.Time{})}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	f := fleet(t, 2)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))
	load(t, f[0], 1)
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[1] {
		t.Errorf("routed to %s, want the only usable backend", got.URL)
	}
}

// The gauge must return to zero on release however many times release fires.
func TestActiveTokensLeaveOnRelease(t *testing.T) {
	r := registry.New(registry.Options{})
	b, err := r.Add(registry.Spec{URL: "http://w:8000"})
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	l := lease.Acquire(b)
	l.StartActive(500)
	l.StartActive(500)
	if got := b.ActiveTokens(); got != 500 {
		t.Fatalf("active = %d, want 500", got)
	}
	l.Release()
	l.Release()
	if got := b.ActiveTokens(); got != 0 {
		t.Errorf("active = %d, want 0 after release", got)
	}
	l.StartActive(500)
	if got := b.ActiveTokens(); got != 0 {
		t.Errorf("active = %d, want 0 when started after release", got)
	}
}
