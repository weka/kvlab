package affinity

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/clock"
	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/registry"
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

// In-flight prefill load counts on top of active load, so a holder busy
// prefilling loses a request to an idle backend sooner.
func TestCostRoutingCountsPrefillLoad(t *testing.T) {
	p := newCostPolicy(t)
	f := fleet(t, 2)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))

	f[0].AddPrefillTokens(384)
	f[0].AddActiveTokens(384)
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[0] {
		t.Fatalf("routed to %s, want the holder while 384+384 < move cost 768", got.URL)
	}
	f[0].AddPrefillTokens(1)
	got, err = p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[1] {
		t.Errorf("routed to %s, want the idle backend once prefill pushes load past move cost", got.URL)
	}
}

// Select records the uncached tokens on the chosen backend for the proxy.
func TestCostRoutingRecordsPrefillEstimate(t *testing.T) {
	p := newCostPolicy(t)
	f := fleet(t, 2)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))
	rr := req(u)
	got, err := p.Select(context.Background(), f, rr)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[0] || rr.Prefill == nil || rr.Prefill.Backend != f[0] || rr.Prefill.Tokens != 0 {
		t.Errorf("prefill = %+v, want holder with 0 uncached tokens", rr.Prefill)
	}
	rr = req(units(7, 8, 9))
	if _, err := p.Select(context.Background(), f, rr); err != nil {
		t.Fatalf("Select: %v", err)
	}
	if rr.Prefill == nil || rr.Prefill.Tokens != 768 {
		t.Errorf("prefill = %+v, want 768 uncached tokens", rr.Prefill)
	}
}

func TestPrefillTokensLifecycle(t *testing.T) {
	r := registry.New(registry.Options{})
	b, err := r.Add(registry.Spec{URL: "http://w:8000"})
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	// First byte removes the charge; Release afterwards does not subtract again.
	l := lease.Acquire(b)
	l.StartPrefill(300)
	l.StartPrefill(300)
	if got := b.PrefillTokens(); got != 300 {
		t.Fatalf("prefill = %d, want 300", got)
	}
	l.EndPrefill()
	l.EndPrefill()
	if got := b.PrefillTokens(); got != 0 {
		t.Fatalf("prefill = %d, want 0 after first byte", got)
	}
	l.Release()
	if got := b.PrefillTokens(); got != 0 {
		t.Errorf("prefill = %d, want 0 (no double subtract)", got)
	}

	// No first byte: Release removes it; late starts and ends are no-ops.
	l = lease.Acquire(b)
	l.StartPrefill(200)
	l.Release()
	if got := b.PrefillTokens(); got != 0 {
		t.Errorf("prefill = %d, want 0 after Release without first byte", got)
	}
	l.StartPrefill(200)
	l.EndPrefill()
	if got := b.PrefillTokens(); got != 0 {
		t.Errorf("prefill = %d, want 0 when started after release", got)
	}
}

func newCostSignalPolicy(t *testing.T, c Config) *Policy {
	t.Helper()
	c.CostRouting = true
	c.TailTTL = testTTL
	c.Clock = clock.NewFake(time.Time{})
	p, err := New(c, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// A backend at --max-node-concurrency is excluded even when it holds the whole
// prefix and its cost would otherwise win.
func TestCostRoutingExcludesBackendAtMaxNodeConcurrency(t *testing.T) {
	p := newCostSignalPolicy(t, Config{NodeConcurrency: 2})
	f := fleet(t, 2)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))
	load(t, f[0], 2)
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[1] {
		t.Errorf("routed to %s, want the unsaturated backend", got.URL)
	}
}

// With every backend at the limit, cost routing answers like the ladder: a
// SaturatedError naming the backend to queue on (so the wait queue works), and
// ErrAllBackendsSaturated when no backend can be waited on.
func TestCostRoutingAllSaturatedQueuesOrRejects(t *testing.T) {
	p := newCostSignalPolicy(t, Config{NodeConcurrency: 2})
	f := fleet(t, 2)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))
	load(t, f[0], 2)
	load(t, f[1], 2)
	err := selectSaturated(t, p, f, req(u))
	var sat *policy.SaturatedError
	if !errors.As(err, &sat) {
		t.Fatalf("error = %v, want a *policy.SaturatedError", err)
	}
	if sat.Preferred != f[0] {
		t.Errorf("Preferred = %s, want the holder %s", sat.Preferred.URL, f[0].URL)
	}

	off := newCostSignalPolicy(t, Config{})
	for _, b := range f {
		off.AddBackend(b)
	}
	off.OnRefused(f[0])
	off.OnRefused(f[1])
	if _, err := off.Select(context.Background(), f, req(u)); !errors.Is(err, policy.ErrAllBackendsSaturated) {
		t.Errorf("all refused: err = %v, want ErrAllBackendsSaturated", err)
	}
}

// --hard-ttft-split excludes a slow holder under cost routing.
func TestCostRoutingExcludesBackendOverHardTTFT(t *testing.T) {
	p := newCostSignalPolicy(t, Config{TTFTHard: 2 * time.Second})
	f := fleet(t, 2)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))
	setTTFT(f[0], 5*time.Second)
	f[0].NoteAdmit() // the hard gate closes once a request was admitted over the limit
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got != f[1] {
		t.Errorf("routed to %s, want the backend under the TTFT limit", got.URL)
	}
}

// --rebalance-ratio excludes a backend carrying far more than the fleet minimum.
func TestCostRoutingExcludesImbalancedBackend(t *testing.T) {
	p := newCostSignalPolicy(t, Config{RebalanceRatio: 0.5})
	f := fleet(t, 3)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))
	load(t, f[0], 20)
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got == f[0] {
		t.Errorf("routed to the imbalanced holder %s, want it excluded", got.URL)
	}
}
