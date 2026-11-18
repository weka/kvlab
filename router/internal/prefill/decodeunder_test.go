package prefill

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/weka/kvlab/router/internal/registry"
)

func probeOf(readings map[string]probeReading) DecodeProbe {
	return func(_ context.Context, b *registry.Backend) (float64, float64, error) {
		r := readings[b.URL]
		return r.kv, r.waiting, r.err
	}
}

// underFleet is starveFleet with underutilization on at idleFraction
// (idle below KV 0.6, add at 2s).
func underFleet(t *testing.T, n int, idleFraction float64, readings map[string]probeReading) *fleet {
	t.Helper()
	return pdModeFleet(t, n, Config{
		MaxFleetFraction: 0.8,
		TTFTWindow:       4, TTFTHigh: 5 * time.Second, TTFTLow: time.Second,
		StarveKV: 0.95, StarveWaiting: 1, StarveFraction: 0.5,
		StarvedShrinkGuard: 5 * time.Second, StarvedTTFTHigh: 10 * time.Second,
		IdleKV: 0.6, IdleFraction: idleFraction, UnderutilTTFTHigh: 2 * time.Second,
		DecodeProbe: probeOf(readings),
	})
}

func (f *fleet) signals(span time.Duration) decodeSignals {
	f.m.mu.Lock()
	defer f.m.mu.Unlock()
	return f.m.decodeSignalsLocked(f.clk.Now(), span)
}

func gauge(t *testing.T, g prometheus.Gauge) float64 {
	t.Helper()
	var m dto.Metric
	if err := g.Write(&m); err != nil {
		t.Fatal(err)
	}
	return m.GetGauge().GetValue()
}

func TestDecodeIdleSampleIsTheIdleShareOfDecodeNodes(t *testing.T) {
	readings := map[string]probeReading{}
	f := underFleet(t, 6, 0.5, readings)                          // node 0 prefill, 1-5 decode
	readings[f.bs[1].URL] = probeReading{kv: 0.3, waiting: 0}     // idle
	readings[f.bs[2].URL] = probeReading{kv: 0.59, waiting: 0}    // idle, just below
	readings[f.bs[3].URL] = probeReading{kv: 0.6, waiting: 0}     // at the threshold: busy
	readings[f.bs[4].URL] = probeReading{kv: 0.1, waiting: 1}     // something waits: busy
	readings[f.bs[5].URL] = probeReading{err: errors.New("down")} // unread: not idle
	readings[f.bs[0].URL] = probeReading{}                        // a prefill node is not sampled
	f.m.SampleDecode(context.Background(), f.bs)
	s := f.signals(time.Minute)
	if s.n != 1 || s.idleFrac != 0.4 || s.under || s.starved {
		t.Errorf("signals %+v, want 2 of 5 idle: 0.4, not underutilized", s)
	}
	readings[f.bs[3].URL] = probeReading{kv: 0.2}
	readings[f.bs[4].URL] = probeReading{kv: 0.2}
	f.m.SampleDecode(context.Background(), f.bs)
	if s := f.signals(time.Minute); s.n != 2 || s.idleFrac < 0.599 || s.idleFrac > 0.601 || !s.under {
		t.Errorf("signals %+v, want mean of 0.4 and 0.8: 0.6, underutilized", s)
	}
	f.clk.Advance(time.Millisecond)
	if s := f.signals(0); s.n != 0 || s.under {
		t.Errorf("no sample in the window: %+v, want not underutilized", s)
	}
}

func TestDecodeUnderutilizationDisableSwitch(t *testing.T) {
	for _, frac := range []float64{0, 1.01, 2} {
		readings := map[string]probeReading{}
		f := underFleet(t, 3, frac, readings)
		for _, b := range f.bs {
			readings[b.URL] = probeReading{kv: 0}
		}
		f.m.SampleDecode(context.Background(), f.bs)
		if s := f.signals(time.Minute); s.under || s.idleFrac != 1 {
			t.Errorf("idle fraction %v: signals %+v, want all idle but not underutilized", frac, s)
		}
	}
}

// The rule branches with underutilization, and its precedence: starvation,
// then underutilization, then the normal rules. Nodes 0 and 1 prefill, 2-4
// decode; per decode node a reading.
func TestDecodeUnderutilizationRules(t *testing.T) {
	idle := probeReading{kv: 0.2}
	busy := probeReading{kv: 0.8}
	starved := probeReading{kv: 0.99, waiting: 3}
	for _, c := range []struct {
		name           string
		idleFraction   float64
		decode         [3]probeReading
		ttft           time.Duration
		wantP          int
		action, reason string
	}{
		{"underutilized, slow prefill: add", 0.5, [3]probeReading{idle, idle, busy}, 3 * time.Second, 3, "drain", "underutil_ttft_high"},
		{"underutilized, at the threshold: add", 0.5, [3]probeReading{idle, idle, busy}, 2 * time.Second, 3, "drain", "underutil_ttft_high"},
		{"underutilized, fast prefill: hold, never return", 0.5, [3]probeReading{idle, idle, idle}, 500 * time.Millisecond, 2, "hold", "underutil_hold"},
		{"underutilized, between: hold", 0.5, [3]probeReading{idle, idle, idle}, 1500 * time.Millisecond, 2, "hold", "underutil_hold"},
		{"starved before underutilized", 0.3, [3]probeReading{starved, starved, idle}, 3 * time.Second, 1, "release", "decode_starved_shrink"},
		{"not underutilized: normal low", 0.5, [3]probeReading{idle, busy, busy}, 500 * time.Millisecond, 1, "release", "ttft_low"},
		{"not underutilized: normal between", 0.5, [3]probeReading{idle, busy, busy}, 3 * time.Second, 2, "", ""},
		{"disabled: normal low", 1.5, [3]probeReading{idle, idle, idle}, 500 * time.Millisecond, 1, "release", "ttft_low"},
	} {
		t.Run(c.name, func(t *testing.T) {
			readings := map[string]probeReading{}
			f := underFleet(t, 5, c.idleFraction, readings)
			f.setMode(f.bs[1], Prefill, f.clk.Now().Add(time.Second))
			for i, r := range c.decode {
				readings[f.bs[2+i].URL] = r
			}
			f.m.SampleDecode(context.Background(), f.bs)
			f.ttft(4, c.ttft)
			f.m.Tick(f.bs)
			if got := count(f.modes(), Prefill) + count(f.modes(), Draining); got != c.wantP {
				t.Errorf("%d nodes out of decode, want %d (modes %v)", got, c.wantP, f.modes())
			}
			for _, r := range []string{"underutil_ttft_high", "underutil_hold", "decode_starved_shrink", "ttft_low"} {
				for _, a := range []string{"drain", "hold", "release"} {
					want := 0.0
					if a == c.action && r == c.reason {
						want = 1
					}
					if got := actionCount(t, a, r); got != want {
						t.Errorf("%s{reason=%s} = %v, want %v", a, r, got, want)
					}
				}
			}
			if c.action == "hold" {
				f.m.Tick(f.bs)
				if got := actionCount(t, "hold", "underutil_hold"); got != 1 {
					t.Errorf("hold counted %v times on one window, want 1", got)
				}
			}
		})
	}
}

// An empty prefill-finish window takes no action even underutilized.
func TestDecodeUnderutilizedButEmptyWindowTakesNoAction(t *testing.T) {
	readings := map[string]probeReading{}
	f := underFleet(t, 5, 0.5, readings)
	f.m.SampleDecode(context.Background(), f.bs)
	f.m.Tick(f.bs)
	if got := actionCount(t, "hold", "underutil_hold"); got != 0 || count(f.modes(), Prefill) != 1 {
		t.Errorf("acted on an empty window: hold %v, modes %v", got, f.modes())
	}
}

// underTimeFleet runs the 30s time window with the 10s underutilization
// window; node 0 is prefill, the rest decode.
func underTimeFleet(t *testing.T, n int, readings map[string]probeReading) *fleet {
	return timeFleet(t, n, Config{
		MaxFleetFraction: 0.9,
		StarveKV:         0.95, StarveWaiting: 1, StarveFraction: 0.5,
		StarvedShrinkGuard: 5 * time.Second, StarvedTTFTHigh: 10 * time.Second,
		IdleKV: 0.6, IdleFraction: 0.5, UnderutilTTFTHigh: 2 * time.Second, UnderutilWindowTime: 10 * time.Second,
		DecodeProbe: probeOf(readings),
	})
}

// While underutilized the 10s window is in force: the mean reads the last
// 10s only, and the next action comes 10s after the last, not 30s.
func TestDecodeUnderutilizedSwitchesToTheShortWindow(t *testing.T) {
	for _, c := range []struct {
		name     string
		kv       float64
		wantNext int // drains after the first, ticking every second for 29s
	}{
		{"underutilized: 10s window", 0.2, 2},
		{"busy: 30s window", 0.8, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			readings := map[string]probeReading{}
			f := underTimeFleet(t, 8, readings)
			for _, b := range f.bs {
				readings[b.URL] = probeReading{kv: c.kv}
			}
			f.spread(20, 3*time.Second, 10*time.Second)
			f.m.SampleDecode(context.Background(), f.bs)
			f.m.Tick(f.bs)
			drains := func() float64 {
				return actionCount(t, "drain", "underutil_ttft_high") + actionCount(t, "drain", "ttft_high")
			}
			if drains() != 1 {
				t.Fatalf("first window: %v drains, want 1", drains())
			}
			for range 29 {
				f.spread(2, 3*time.Second, time.Second)
				f.m.SampleDecode(context.Background(), f.bs)
				f.m.Tick(f.bs)
			}
			if got := int(drains()) - 1; got != c.wantNext {
				t.Errorf("%d more drains within 29s, want %d", got, c.wantNext)
			}
		})
	}
}

func TestDecodeUnderutilizedMeanReadsTheShortWindow(t *testing.T) {
	readings := map[string]probeReading{}
	f := underTimeFleet(t, 4, readings)
	f.spread(20, 10*time.Second, 20*time.Second) // older than 10s, slow
	f.spread(20, time.Second, 10*time.Second)    // the last 10s: 1s
	f.clk.Advance(time.Millisecond)
	f.m.SampleDecode(context.Background(), f.bs) // all idle (kv 0)
	f.m.Tick(f.bs)
	if got := gauge(t, f.m.cfg.PDMetrics.TTFTMean); got != 1 {
		t.Errorf("window mean %vs, want 1s: the last 10s only", got)
	}
	if got := actionCount(t, "hold", "underutil_hold"); got != 1 || count(f.modes(), Prefill) != 1 {
		t.Errorf("hold %v, modes %v: want a hold on a 1s mean, no 30s-window drain", got, f.modes())
	}
}

// Each flag is its threshold applied to the fraction published with it, at
// every tick, including right after an action restarted the window.
func TestDecodeSignalGaugesAgree(t *testing.T) {
	readings := map[string]probeReading{}
	f := underFleet(t, 5, 0.5, readings)
	f.setMode(f.bs[1], Prefill, f.clk.Now().Add(time.Second))
	pm := f.m.cfg.PDMetrics
	check := func(step string, starved, under float64) {
		t.Helper()
		sf, s := gauge(t, pm.DecodeStarvedFraction), gauge(t, pm.DecodeStarved)
		uf, u := gauge(t, pm.DecodeUnderutilizedFraction), gauge(t, pm.DecodeUnderutilized)
		if s != boolGauge(sf >= 0.5) || u != boolGauge(uf >= 0.5) {
			t.Errorf("%s: starved %v at fraction %v, underutilized %v at fraction %v: flags disagree", step, s, sf, u, uf)
		}
		if s != starved || u != under {
			t.Errorf("%s: starved %v underutilized %v, want %v %v", step, s, u, starved, under)
		}
	}
	set := func(r probeReading) {
		for _, b := range f.bs {
			readings[b.URL] = r
		}
		f.m.SampleDecode(context.Background(), f.bs)
	}
	set(probeReading{kv: 0.99, waiting: 5})
	f.m.Tick(f.bs)
	check("starved, no request", 1, 0)
	f.ttft(4, 3*time.Second)
	f.m.Tick(f.bs) // publishes what the starved shrink read, then restarts the window
	check("the action's tick", 1, 0)
	if got := actionCount(t, "release", "decode_starved_shrink"); got != 1 {
		t.Fatalf("starved shrink %v, want 1", got)
	}
	f.m.Tick(f.bs)
	check("after the action: no sample left", 0, 0)
	if gauge(t, pm.DecodeStarvedFraction) != 0 {
		t.Error("fraction kept a sample from before the action")
	}
	set(probeReading{kv: 0.1})
	f.m.Tick(f.bs)
	check("idle", 0, 1)
	set(probeReading{kv: 0.99, waiting: 5})
	f.m.Tick(f.bs)
	check("half idle, half starved", 1, 1)
}
