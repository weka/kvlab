package affinity

import (
	"context"
	"testing"
	"time"

	"github.com/weka/kvlab/kvcache"
	"github.com/weka/kvlab/router/internal/clock"
	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/registry"
)

func newRelPolicy(t *testing.T, relTTFT, relDec bool, softTTFT time.Duration) *Policy {
	t.Helper()
	p, err := New(Config{
		RelativeTTFT:   relTTFT,
		RelativeDecode: relDec,
		TTFTSoft:       softTTFT,
		SplitGuard:     DefaultSplitGuard,
		TailTTL:        testTTL,
		Clock:          clock.NewFake(time.Time{}),
	}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func addDecode(b *registry.Backend, n int64) { b.AddDecodeTokens(n) }

// pickN selects n times and returns the distinct backends chosen.
func pickN(t *testing.T, p *Policy, f []*registry.Backend, u []kvcache.Unit, est int64, n int) map[*registry.Backend]bool {
	t.Helper()
	got := map[*registry.Backend]bool{}
	for i := 0; i < n; i++ {
		rr := req(u)
		rr.EstTokens = est
		b, err := p.Select(context.Background(), f, rr)
		if err != nil {
			t.Fatalf("Select: %v", err)
		}
		got[b] = true
	}
	return got
}

// ---------------------------------------------------------------- relative TTFT

// 1s vs 2s and 10s vs 20s must behave identically: the comparison is pure, no
// absolute constant is involved anywhere.
func TestRelativeTTFTIsScaleInvariant(t *testing.T) {
	for _, scale := range []time.Duration{time.Second, 10 * time.Second} {
		p := newRelPolicy(t, true, false, 0)
		f, u := softFleet(t, p, 0, 0)
		setTTFT(f[0], 2*scale) // holder
		setTTFT(f[1], scale)
		setTTFT(f[2], scale)
		before := counter(t, metrics.SplitTriggers.WithLabelValues("relative_ttft"))
		for b := range pickN(t, p, f, u, 0, 4) {
			if b == f[0] {
				t.Errorf("scale %v: holder slower than every peer was not split off", scale)
			}
		}
		if d := counter(t, metrics.SplitTriggers.WithLabelValues("relative_ttft")) - before; d < 1 {
			t.Errorf("scale %v: router_split_trigger_total{relative_ttft} did not move", scale)
		}
	}
}

func TestRelativeTTFTEqualP90StaysOnHolder(t *testing.T) {
	p := newRelPolicy(t, true, false, 0)
	f, u := softFleet(t, p, 0, 0)
	for _, b := range f {
		setTTFT(b, time.Second) // identical: not worse than its peers
	}
	for b := range pickN(t, p, f, u, 0, 4) {
		if b != f[0] {
			t.Errorf("equal p90: the holder must keep the request, got %s", b.URL)
		}
	}
}

// A backend with too few samples reads 0 and is an eligible target, so it can
// collect samples.
func TestRelativeTTFTColdTargetIsEligible(t *testing.T) {
	p := newRelPolicy(t, true, false, 0)
	f, u := softFleet(t, p, 0, 0)
	setTTFT(f[0], 2*time.Second)
	setTTFT(f[1], 2*time.Second) // not eligible: not below the holder
	// f[2] is cold: p90 == 0
	for b := range pickN(t, p, f, u, 0, 4) {
		if b != f[2] {
			t.Errorf("want the cold node, got %s", b.URL)
		}
	}
}

// The cache split guard still gates a relative split: if no target is clearly
// below the holder's in-flight, the holder keeps the request.
func TestRelativeTTFTGuardBlockedStaysOnHolder(t *testing.T) {
	p := newRelPolicy(t, true, false, 0)
	f, u := softFleet(t, p, 10, 9) // peers only 10% below the holder in-flight: guard needs 20%
	setTTFT(f[0], 3*time.Second)
	setTTFT(f[1], time.Second)
	setTTFT(f[2], time.Second)
	for b := range pickN(t, p, f, u, 0, 4) {
		if b != f[0] {
			t.Errorf("guard must keep the holder, got %s", b.URL)
		}
	}
}

// ---------------------------------------------------------------- relative decode

func TestRelativeDecodeSplitsOnImbalance(t *testing.T) {
	p := newRelPolicy(t, false, true, 0)
	f, u := softFleet(t, p, 0, 0)
	addDecode(f[0], 100_000)
	addDecode(f[1], 10_000)
	addDecode(f[2], 10_000)
	before := counter(t, metrics.SplitTriggers.WithLabelValues("relative_decode"))
	for b := range pickN(t, p, f, u, 1000, 4) {
		if b == f[0] {
			t.Errorf("heavily loaded holder was not split off")
		}
	}
	if d := counter(t, metrics.SplitTriggers.WithLabelValues("relative_decode")) - before; d < 1 {
		t.Errorf("router_split_trigger_total{relative_decode} did not move")
	}
}

// Anti-flap: the move must not just invert the imbalance. No constant floor is
// involved; the request's own tokens are part of the comparison.
func TestRelativeDecodeAntiFlap(t *testing.T) {
	cases := []struct {
		name               string
		holder, other, est int64
		wantSplit          bool
	}{
		{"would end up equal", 10_000, 9_000, 1_000, false},
		{"would end up worse", 10_000, 9_500, 1_000, false},
		{"low load, one-token request", 1, 0, 1, false}, // 0+1 < 1 is false
		{"very large request would invert", 10_000, 0, 100_000, false},
		{"stays lighter after the request", 10_000, 8_000, 1_000, true},
		{"clear imbalance", 100_000, 10_000, 1_000, true},
	}
	for _, c := range cases {
		p := newRelPolicy(t, false, true, 0)
		f, u := softFleet(t, p, 0, 0)
		addDecode(f[0], c.holder)
		addDecode(f[1], c.other)
		addDecode(f[2], c.other)
		for b := range pickN(t, p, f, u, c.est, 4) {
			if split := b != f[0]; split != c.wantSplit {
				t.Errorf("%s: split=%v, want %v", c.name, split, c.wantSplit)
			}
		}
	}
}

// Eligibility is measured AFTER the target takes the request.
func TestRelativeDecodeTargetEligibleAfterRequest(t *testing.T) {
	p := newRelPolicy(t, false, true, 0)
	f, u := softFleet(t, p, 0, 0)
	addDecode(f[0], 100_000)
	addDecode(f[1], 60_000) // 60k+40k = 100k, not below 100k: would end up equal
	addDecode(f[2], 1_000)  // 1k+40k = 41k < 100k
	for b := range pickN(t, p, f, u, 40_000, 5) {
		if b != f[2] {
			t.Errorf("want only the target that stays lighter after the request, got %s", b.URL)
		}
	}
}

// ---------------------------------------------------------------- composition

// With both relative signals on, a target must pass both, and soft TTFT's
// eligibility on top.
func TestRelativeSignalsComposeTargetMustPassBoth(t *testing.T) {
	p := newRelPolicy(t, true, true, 0)
	f := fleet(t, 4)
	for _, b := range f {
		p.AddBackend(b)
	}
	u := units(1, 2, 3)
	p.Commit(f[0], req(u))

	setTTFT(f[0], 2*time.Second)
	addDecode(f[0], 100_000)
	setTTFT(f[1], time.Second)   // ttft ok ...
	addDecode(f[1], 99_500)      // ... decode not: 99.5k+1k >= 100k
	setTTFT(f[2], 2*time.Second) // decode ok ...
	addDecode(f[2], 1_000)       // ... ttft not: not below the holder
	setTTFT(f[3], time.Second)
	addDecode(f[3], 1_000)
	for b := range pickN(t, p, f, u, 1000, 5) {
		if b != f[3] {
			t.Errorf("only the node passing both is a valid target, got %s", b.URL)
		}
	}
}

func TestRelativeOffLeavesHolderAlone(t *testing.T) {
	p := newRelPolicy(t, false, false, 0)
	f, u := softFleet(t, p, 0, 0)
	setTTFT(f[0], time.Hour)
	addDecode(f[0], 1_000_000)
	for b := range pickN(t, p, f, u, 1000, 4) {
		if b != f[0] {
			t.Errorf("both relative signals off: holder must keep the request")
		}
	}
}
