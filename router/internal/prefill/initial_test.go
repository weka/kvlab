package prefill

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/clock"
	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/registry"
)

// newFleetAt is newFleet over the given URLs.
func newFleetAt(t *testing.T, urls []string, cfg Config) *fleet {
	t.Helper()
	reg := registry.New(registry.Options{})
	f := &fleet{clk: clock.NewFake(time.Time{})}
	for _, u := range urls {
		b, err := reg.Add(registry.Spec{URL: u, Prov: registry.ProvStatic, Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
		f.bs = append(f.bs, b)
	}
	cfg.Clock = f.clk
	cfg.PoolName = t.Name()
	f.m = New(cfg)
	return f
}

func TestInitialNodesArePrefillAtFirstSight(t *testing.T) {
	var buf bytes.Buffer
	f := newFleet(t, 8, Config{InitialNodes: 3, ConvertAfter: 10 * time.Second, DrainMinInterval: time.Minute,
		Log: slog.New(slog.NewTextHandler(&buf, nil))})
	f.m.Tick(f.bs)
	want := []Mode{Prefill, Prefill, Prefill, Decode, Decode, Decode, Decode, Decode}
	for i, m := range f.modes() {
		if m != want[i] {
			t.Fatalf("modes = %v, want %v", f.modes(), want)
		}
	}
	if got := f.m.State().Conversions; got != 0 {
		t.Errorf("conversions = %d, want 0: no node was drained", got)
	}
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("log = %q, want 3 lines", buf.String())
	}
	last := lines[2]
	for _, s := range []string{`msg="draining node is now a prefill node"`, "reason=initial", "prefill_nodes=3", "draining_nodes=0", "decode_nodes=5", "backend=http://n2:8000"} {
		if !strings.Contains(last, s) {
			t.Errorf("log line %q lacks %q", last, s)
		}
	}
	f.m.Tick(f.bs)
	if got := count(f.modes(), Prefill); got != 3 {
		t.Errorf("prefill nodes after a second tick = %d, want 3", got)
	}
	// A rejected prefill goes straight to one of them: no drain.
	j, err := f.m.Reserve(f.bs, 50_000, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer j.Done()
	if f.m.ModeOf(j.Backend) != Prefill || f.m.State().Conversions != 0 {
		t.Errorf("job on %s (%v), conversions %d: want an initial node, no drain", j.Backend.URL, f.m.ModeOf(j.Backend), f.m.State().Conversions)
	}
}

func TestInitialNodesBeforeTheFirstTickServeTheFirstReject(t *testing.T) {
	f := newFleet(t, 4, Config{InitialNodes: 1})
	j, err := f.m.Reserve(f.bs, 50_000, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer j.Done()
	if j.Backend != f.bs[0] || f.m.ModeOf(f.bs[0]) != Prefill || f.m.State().Conversions != 0 {
		t.Errorf("job on %s, modes %v: want the initial node n0", j.Backend.URL, f.modes())
	}
}

func TestInitialNodesSpreadAcrossHosts(t *testing.T) {
	f := newFleetAt(t, []string{
		"http://10.0.0.1:8000", "http://10.0.0.1:8001", "http://10.0.0.1:8002",
		"http://10.0.0.2:8000", "http://10.0.0.2:8001", "http://10.0.0.2:8002",
		"http://10.0.0.3:8000", "http://10.0.0.3:8001", "http://10.0.0.3:8002",
	}, Config{InitialNodes: 4})
	f.m.Tick(f.bs)
	got := f.modes()
	want := []Mode{Prefill, Prefill, Decode, Prefill, Decode, Decode, Prefill, Decode, Decode}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("modes = %v, want %v", got, want)
		}
	}
}

func TestInitialNodesReachTheCountAsPodsAppear(t *testing.T) {
	f := newFleet(t, 10, Config{InitialNodes: 3})
	// One pod: the last decode node is never converted.
	f.m.Tick(f.bs[:1])
	if got := count(f.modes(), Prefill); got != 0 {
		t.Fatalf("prefill nodes with one pod = %d, want 0", got)
	}
	f.m.Tick(f.bs[:4]) // cap floor(4*0.5)=2
	if got := count(f.modes(), Prefill); got != 2 {
		t.Fatalf("prefill nodes with four pods = %d, want 2", got)
	}
	f.m.Tick(f.bs)
	f.m.Tick(f.bs)
	if got := count(f.modes(), Prefill); got != 3 {
		t.Fatalf("prefill nodes with ten pods = %d, want 3", got)
	}
}

func TestInitialNodesCappedByFleetFraction(t *testing.T) {
	f := newFleet(t, 6, Config{InitialNodes: 10, MaxFleetFraction: 0.5})
	f.m.Tick(f.bs)
	f.m.Tick(f.bs)
	if got := count(f.modes(), Prefill); got != 3 {
		t.Errorf("prefill nodes = %d, want floor(6*0.5)=3", got)
	}
}

func TestInitialNodesSkipABusyNode(t *testing.T) {
	f := newFleet(t, 4, Config{InitialNodes: 1})
	l := lease.Acquire(f.bs[0])
	f.m.Tick(f.bs)
	if f.m.ModeOf(f.bs[0]) != Decode || f.m.ModeOf(f.bs[1]) != Prefill {
		t.Errorf("modes = %v, want the idle n1 taken, the busy n0 left in decode", f.modes())
	}
	l.Release()
}

func TestInitialNodesReleasedByIdleLikeAnyOther(t *testing.T) {
	f := newFleet(t, 4, Config{InitialNodes: 3, ReleaseIdle: 30 * time.Second})
	f.m.Tick(f.bs) // cap 2
	if got := count(f.modes(), Prefill); got != 2 {
		t.Fatalf("prefill nodes = %d, want 2", got)
	}
	// No job yet: the idle clock does not run, however long the wait.
	for range 3 {
		f.clk.Advance(time.Minute)
		f.m.Tick(f.bs)
	}
	if got := count(f.modes(), Prefill); got != 2 {
		t.Fatalf("modes = %v: an initial node with no job was released", f.modes())
	}
	j, err := f.m.Reserve(f.bs, 50_000, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.m.WaitReady(context.Background(), j); err != nil {
		t.Fatal(err)
	}
	served := j.Backend
	j.Done()
	f.clk.Advance(29 * time.Second)
	f.m.Tick(f.bs)
	if f.m.ModeOf(served) != Prefill {
		t.Fatalf("%s released before the grace", served.URL)
	}
	f.clk.Advance(time.Second)
	f.m.Tick(f.bs)
	f.m.Tick(f.bs)
	if f.m.ModeOf(served) != Decode {
		t.Fatalf("modes = %v, want %s released after ReleaseIdle", f.modes(), served.URL)
	}
	if got := f.m.State().Releases; got != 1 {
		t.Errorf("releases = %d, want 1", got)
	}
	// The other initial node, never used, stays; nothing new is taken.
	f.clk.Advance(time.Minute)
	f.m.Tick(f.bs)
	if p, d := count(f.modes(), Prefill), count(f.modes(), Decode); p != 1 || d != 3 {
		t.Errorf("modes = %v, want the unused initial node only", f.modes())
	}
}

func TestNoInitialNodesByDefault(t *testing.T) {
	f := newFleet(t, 4, Config{})
	f.m.Tick(f.bs)
	if got := count(f.modes(), Decode); got != 4 {
		t.Errorf("modes = %v, want all decode", f.modes())
	}
}

func TestInitialNodesIgnoredUnderPD(t *testing.T) {
	f := newFleet(t, 4, Config{PD: true, InitialNodes: 2})
	if f.m.initialLeft != 0 {
		t.Errorf("initialLeft = %d under PD, want 0", f.m.initialLeft)
	}
}
