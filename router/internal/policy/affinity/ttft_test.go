package affinity

import (
	"context"
	"testing"
	"time"

	"github.com/weka/wekai/router/internal/clock"
	"github.com/weka/wekai/router/internal/registry"
)

func newTTFTPolicy(t *testing.T, hard, soft time.Duration) *Policy {
	t.Helper()
	p, err := New(Config{
		TTFTHard:   hard,
		TTFTSoft:   soft,
		SplitGuard: DefaultSplitGuard,
		TailTTL:    testTTL,
		Clock:      clock.NewFake(time.Time{}),
	}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func setTTFT(b *registry.Backend, d time.Duration) { b.ObserveTTFT(d) }

func TestTTFTEWMA(t *testing.T) {
	b := fleet(t, 1)[0]
	if b.TTFT() != 0 {
		t.Fatalf("no samples should read 0, got %v", b.TTFT())
	}
	b.ObserveTTFT(time.Second)
	if b.TTFT() != time.Second {
		t.Fatalf("first sample seeds, got %v", b.TTFT())
	}
	b.ObserveTTFT(2 * time.Second)
	if got := b.TTFT(); got <= time.Second || got >= 2*time.Second {
		t.Fatalf("EWMA should land between samples, got %v", got)
	}
}

func TestHardTTFTExcludesSlowBackend(t *testing.T) {
	p := newTTFTPolicy(t, 5*time.Second, 0)
	f, u := softFleet(t, p, 0, 0)
	setTTFT(f[0], 10*time.Second) // the holder is over the hard limit
	f[0].NoteAdmit()              // a request was admitted over the limit: gate closes
	for i := 0; i < 5; i++ {
		got, err := p.Select(context.Background(), f, req(u))
		if err != nil {
			t.Fatalf("Select: %v", err)
		}
		if got == f[0] {
			t.Fatalf("routed to the over-threshold backend")
		}
	}
}

func TestHardTTFTGateReopensOnCompletion(t *testing.T) {
	p := newTTFTPolicy(t, 5*time.Second, 0)
	f, u := softFleet(t, p, 0, 0)
	setTTFT(f[0], 10*time.Second)
	// Over the limit but no request admitted yet: not excluded (no deadlock).
	if got, err := p.Select(context.Background(), f, req(u)); err != nil || got != f[0] {
		t.Fatalf("open gate should still admit the holder, got %v %v", got, err)
	}
	f[0].NoteAdmit()
	if got, _ := p.Select(context.Background(), f, req(u)); got == f[0] {
		t.Fatal("closed gate should exclude the holder")
	}
	f[0].NoteRelease() // any completion reopens
	if got, err := p.Select(context.Background(), f, req(u)); err != nil || got != f[0] {
		t.Fatalf("gate should reopen after a completion, got %v %v", got, err)
	}
}

func TestSoftTTFTSplitsWithoutHard(t *testing.T) {
	p := newTTFTPolicy(t, 0, 2*time.Second)
	f, u := softFleet(t, p, 0, 0)
	setTTFT(f[0], 3*time.Second) // past soft, no hard configured
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got == f[0] {
		t.Errorf("holder past soft TTFT should have been split off")
	}
}

func TestSoftTTFTBelowThresholdIsCacheHit(t *testing.T) {
	p := newTTFTPolicy(t, 0, 2*time.Second)
	f, u := softFleet(t, p, 0, 0)
	setTTFT(f[0], time.Second)
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil || got != f[0] {
		t.Errorf("got %v, %v; want the holder", got, err)
	}
}

func TestSoftTTFTAllOverSoftStaysOnHolder(t *testing.T) {
	p := newTTFTPolicy(t, 0, 2*time.Second)
	f, u := softFleet(t, p, 0, 0)
	for _, b := range f {
		setTTFT(b, 3*time.Second)
	}
	got, err := p.Select(context.Background(), f, req(u))
	if err != nil || got != f[0] {
		t.Errorf("every node over soft: must stay on the holder, got %v %v", got, err)
	}
}

func TestSoftTTFTSplitGoesOnlyToUnderSoftNode(t *testing.T) {
	p := newTTFTPolicy(t, 0, 2*time.Second)
	f, u := softFleet(t, p, 0, 0)
	setTTFT(f[0], 3*time.Second)
	setTTFT(f[1], 3*time.Second)
	setTTFT(f[2], time.Second) // the only eligible target
	for i := 0; i < 5; i++ {
		if got, err := p.Select(context.Background(), f, req(u)); err != nil || got != f[2] {
			t.Fatalf("want the under-soft node, got %v %v", got, err)
		}
	}
}

func TestSoftTTFTNeverSpillsOntoOverSoftNode(t *testing.T) {
	p := newTTFTPolicy(t, 0, 2*time.Second)
	f, u := softFleet(t, p, 0, 0)
	setTTFT(f[0], 3*time.Second)
	setTTFT(f[1], 3*time.Second) // ineligible
	for i := 0; i < 5; i++ {
		got, err := p.Select(context.Background(), f, req(u))
		if err != nil || got == f[1] {
			t.Fatalf("spilled onto an over-soft node: %v %v", got, err)
		}
	}
}

func TestTTFTOffLeavesSignalsUnchanged(t *testing.T) {
	p := newTTFTPolicy(t, 0, 0)
	for _, n := range signalNames(p) {
		if n == "ttft" {
			t.Errorf("ttft signal registered with the flag off")
		}
	}
}
