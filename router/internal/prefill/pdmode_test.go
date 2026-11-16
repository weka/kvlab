package prefill

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/registry"
)

// pdModeFleet runs the manager as dynamic P/D with bs[0] as the floor.
func pdModeFleet(t *testing.T, n int, cfg Config) *fleet {
	t.Helper()
	f := newFleet(t, n, Config{})
	cfg.PD = true
	cfg.Floor = []string{f.bs[0].URL + "/"} // normalised like the CLI's
	cfg.Clock, cfg.PoolName = f.clk, t.Name()
	f.m = New(cfg)
	return f
}

func names(bs []*registry.Backend) []string {
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = b.URL
	}
	return out
}

func TestPDModeFloorIsPrefillBeforeItIsSeenAndIsNeverReleased(t *testing.T) {
	for _, idle := range []time.Duration{0, time.Second} {
		f := pdModeFleet(t, 3, Config{ReleaseIdle: idle})
		if !f.m.IsPrefill(f.bs[0]) {
			t.Fatal("floor node is not prefill before the manager met it")
		}
		if got := f.m.DecodeNodes(f.bs); len(got) != 2 || got[0] == f.bs[0] || got[1] == f.bs[0] {
			t.Fatalf("decode candidates = %v, want the two non-floor nodes", names(got))
		}
		j, err := f.m.Reserve(f.bs, 10, nil)
		if err != nil || j.Backend != f.bs[0] {
			t.Fatalf("first leg on %v (%v), want the floor node", j, err)
		}
		if err := f.m.WaitReady(context.Background(), j); err != nil {
			t.Fatal(err)
		}
		j.Done()
		f.clk.Advance(time.Hour)
		f.m.releaseExpired()
		if f.m.ModeOf(f.bs[0]) != Prefill {
			t.Errorf("release-idle %v: floor node mode = %v after an idle hour, want prefill", idle, f.m.ModeOf(f.bs[0]))
		}
	}
}

func TestPDModeNeverReturnsAPrefillOrDrainingNodeForDecode(t *testing.T) {
	f := pdModeFleet(t, 3, Config{MaxFleetFraction: 0.7})
	// Only the floor node is healthy: the reject mode would serve the
	// request on it rather than refuse; P/D must not place a D leg there.
	if got := f.m.DecodeNodes(f.bs[:1]); len(got) != 0 {
		t.Fatalf("decode candidates = %v, want none", names(got))
	}
	// Fill the floor so the next leg drains a decode node.
	defer lease.Acquire(f.bs[1]).Release()
	defer lease.Acquire(f.bs[2]).Release()
	defer lease.Acquire(f.bs[2]).Release()
	j1, _ := f.m.Reserve(f.bs, DefaultNodeBudgetTokens, nil)
	defer j1.Done()
	j2, err := f.m.Reserve(f.bs, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer j2.Done()
	if j2.Backend != f.bs[1] || f.m.ModeOf(f.bs[1]) != Draining {
		t.Fatalf("leg on %s in mode %v, want node 1 draining", j2.Backend.URL, f.m.ModeOf(f.bs[1]))
	}
	if got := f.m.DecodeNodes(f.bs); len(got) != 1 || got[0] != f.bs[2] {
		t.Errorf("decode candidates = %v, want node 2 only", names(got))
	}
}

// The drain is the reject mode's: a node drained into prefill runs no leg
// until the decodes it had have finished.
func TestPDModeDrainedNodeTakesLegsOnlyAfterItsDecodesFinish(t *testing.T) {
	f := pdModeFleet(t, 3, Config{MaxFleetFraction: 0.7})
	decode := lease.Acquire(f.bs[1])
	defer lease.Acquire(f.bs[2]).Release()
	defer lease.Acquire(f.bs[2]).Release()
	j1, _ := f.m.Reserve(f.bs, DefaultNodeBudgetTokens, nil)
	defer j1.Done()
	j, err := f.m.Reserve(f.bs, 10, nil)
	if err != nil || j.Backend != f.bs[1] {
		t.Fatalf("Reserve = %v, %v; want node 1", j, err)
	}
	ready := make(chan error, 1)
	go func() { ready <- f.m.WaitReady(context.Background(), j) }()
	select {
	case <-ready:
		t.Fatal("leg started while the drained node still ran a decode")
	case <-time.After(60 * time.Millisecond):
	}
	decode.Release()
	deadline := time.After(2 * time.Second)
	for {
		select {
		case err := <-ready:
			if err != nil {
				t.Fatal(err)
			}
			if f.m.ModeOf(f.bs[1]) != Prefill {
				t.Errorf("mode = %v after the drain, want prefill", f.m.ModeOf(f.bs[1]))
			}
			j.Done()
			return
		case <-deadline:
			t.Fatal("leg never started after the drain")
		default:
			f.clk.Advance(drainPoll)
			time.Sleep(time.Millisecond)
		}
	}
}

func TestPDModeConversionAndReleaseAreLogged(t *testing.T) {
	var buf bytes.Buffer
	f := pdModeFleet(t, 3, Config{NodeBudgetTokens: 100, MaxFleetFraction: 0.7, Log: slog.New(slog.NewTextHandler(&buf, nil))})
	j1, _ := f.m.Reserve(f.bs, 100, nil)
	_ = f.m.WaitReady(context.Background(), j1)
	j2, _ := f.m.Reserve(f.bs, 10, nil)
	_ = f.m.WaitReady(context.Background(), j2)
	j2.Done()
	j1.Done()
	out := buf.String()
	for _, want := range []string{
		`msg="decode node draining for prefill"`, "reason=no_prefill_room", "job_tokens=10", "node_budget_tokens=100",
		"prefill_pending_tokens=100", `msg="prefill node released to decode"`, "reason=idle",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("log lacks %q:\n%s", want, out)
		}
	}
	if strings.Count(out, "released to decode") != 1 {
		t.Errorf("want exactly one release (the floor stays):\n%s", out)
	}
}
