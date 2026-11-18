package prefill

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"

	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/registry"
)

type probeReading struct {
	kv, waiting float64
	err         error
}

// starveFleet runs PD with a count TTFT window of 4 (high 5s, low 1s), the
// starvation rules at their defaults, and a probe that reads readings.
func starveFleet(t *testing.T, n int, readings map[string]probeReading) *fleet {
	t.Helper()
	return pdModeFleet(t, n, Config{
		MaxFleetFraction: 0.8,
		TTFTWindow:       4, TTFTHigh: 5 * time.Second, TTFTLow: time.Second,
		StarveKV: 0.95, StarveWaiting: 1, StarveFraction: 0.5,
		StarvedShrinkGuard: 5 * time.Second, StarvedTTFTHigh: 10 * time.Second,
		DecodeProbe: func(_ context.Context, b *registry.Backend) (float64, float64, error) {
			r := readings[b.URL]
			return r.kv, r.waiting, r.err
		},
	})
}

func (f *fleet) starved() (bool, float64, int) {
	f.m.mu.Lock()
	defer f.m.mu.Unlock()
	return f.m.decodeStarvedLocked(f.clk.Now())
}

func TestDecodeStarvationSampleIsTheStarvedShareOfDecodeNodes(t *testing.T) {
	readings := map[string]probeReading{}
	f := starveFleet(t, 5, readings)                              // node 0 prefill, 1-4 decode
	readings[f.bs[1].URL] = probeReading{kv: 0.97, waiting: 2}    // starved
	readings[f.bs[2].URL] = probeReading{kv: 0.95, waiting: 1}    // starved, at both thresholds
	readings[f.bs[3].URL] = probeReading{kv: 0.99, waiting: 0}    // full but nothing waiting
	readings[f.bs[4].URL] = probeReading{err: errors.New("down")} // unread: not starved
	readings[f.bs[0].URL] = probeReading{kv: 1, waiting: 9}       // a prefill node is not sampled
	f.m.SampleDecode(context.Background(), f.bs)
	starved, frac, n := f.starved()
	if n != 1 || frac != 0.5 || !starved {
		t.Errorf("starved %v fraction %v over %d samples, want 2 of 4 decode nodes: 0.5, starved", starved, frac, n)
	}
}

func TestDecodeStarvationWindowAveragesPrunesAndRestarts(t *testing.T) {
	readings := map[string]probeReading{}
	f := starveFleet(t, 3, readings)
	if starved, _, n := f.starved(); starved || n != 0 {
		t.Fatalf("no sample: starved %v over %d, want not starved", starved, n)
	}
	all := func(kv float64) {
		for _, b := range f.bs {
			readings[b.URL] = probeReading{kv: kv, waiting: 3}
		}
		f.m.SampleDecode(context.Background(), f.bs)
	}
	all(0.99) // 1.0
	f.clk.Advance(5 * time.Second)
	all(0.1) // 0.0
	f.clk.Advance(5 * time.Second)
	all(0.1) // 0.0
	if starved, frac, n := f.starved(); starved || n != 3 || frac < 0.33 || frac > 0.34 {
		t.Fatalf("starved %v fraction %v over %d, want 1/3 over 3 samples: not starved", starved, frac, n)
	}
	f.clk.Advance(21 * time.Second) // the first sample is now older than the 30s window
	if _, frac, n := f.starved(); n != 2 || frac != 0 {
		t.Errorf("fraction %v over %d after the first sample aged out, want 0 over 2", frac, n)
	}
	all(0.99)
	f.m.mu.Lock()
	f.m.restartWindowLocked(f.clk.Now()) // an action
	f.m.mu.Unlock()
	if _, _, n := f.starved(); n != 0 {
		t.Errorf("%d samples after an action, want the window restarted", n)
	}
}

func TestHTTPDecodeProbeReadsKVUsageAndWaiting(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`# HELP vllm:kv_cache_usage_perc x
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.5
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.97
vllm:num_requests_waiting{engine="0",model_name="m"} 2
vllm:num_requests_waiting{engine="1",model_name="m"} 1
vllm:num_requests_running{engine="0",model_name="m"} 30
`))
	}))
	defer srv.Close()
	kv, waiting, err := HTTPDecodeProbe(nil)(context.Background(), &registry.Backend{URL: srv.URL})
	if err != nil || kv != 0.97 || waiting != 3 {
		t.Errorf("probe = %v, %v, %v; want the highest engine's 0.97 and 3 waiting", kv, waiting, err)
	}
	empty := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	defer empty.Close()
	if _, _, err := HTTPDecodeProbe(nil)(context.Background(), &registry.Backend{URL: empty.URL}); err == nil {
		t.Error("a node without KV usage read as a reading")
	}
}

func actionCount(t *testing.T, action, reason string) float64 {
	t.Helper()
	var m dto.Metric
	_ = metrics.PDProvisionActions.WithLabelValues(t.Name(), action, reason).Write(&m)
	return m.GetCounter().GetValue()
}

// The rule branches. Each case starts with nodes 0 and 1 prefill, 2-4 decode,
// a full window of prefill-finish samples, and a starved or idle decode side.
func TestDecodeStarvationRules(t *testing.T) {
	for _, c := range []struct {
		name           string
		starved        bool
		ttft           time.Duration
		wantP          int
		action, reason string
	}{
		{"not starved, high: add", false, 5 * time.Second, 3, "drain", "ttft_high"},
		{"not starved, low: return", false, time.Second, 1, "release", "ttft_low"},
		{"not starved, between: nothing", false, 3 * time.Second, 2, "", ""},
		{"starved, below the guard: return", true, 3 * time.Second, 1, "release", "decode_starved_shrink"},
		{"starved, between guard and high: hold", true, 7 * time.Second, 2, "hold", "decode_starved_hold"},
		{"starved, very slow prefill: add", true, 10 * time.Second, 3, "drain", "decode_starved_ttft_high"},
	} {
		t.Run(c.name, func(t *testing.T) {
			readings := map[string]probeReading{}
			f := starveFleet(t, 5, readings)
			f.setMode(f.bs[1], Prefill, f.clk.Now().Add(time.Second))
			kv := 0.2
			if c.starved {
				kv = 0.99
			}
			for _, b := range f.bs {
				readings[b.URL] = probeReading{kv: kv, waiting: 4}
			}
			f.m.SampleDecode(context.Background(), f.bs)
			f.ttft(4, c.ttft)
			f.m.Tick(f.bs)
			if got := count(f.modes(), Prefill) + count(f.modes(), Draining); got != c.wantP {
				t.Errorf("%d nodes out of decode, want %d (modes %v)", got, c.wantP, f.modes())
			}
			if c.action != "" {
				if got := actionCount(t, c.action, c.reason); got != 1 {
					t.Errorf("%s{reason=%s} = %v, want 1", c.action, c.reason, got)
				}
			}
			if c.action == "hold" {
				// A hold takes the window: no second decision on it.
				f.m.Tick(f.bs)
				if got := actionCount(t, "hold", "decode_starved_hold"); got != 1 {
					t.Errorf("hold counted %v times on one window, want 1", got)
				}
			}
		})
	}
}

// An empty prefill-finish window takes no action even with the decode side
// starved.
func TestDecodeStarvedButEmptyWindowTakesNoAction(t *testing.T) {
	readings := map[string]probeReading{}
	f := starveFleet(t, 5, readings)
	f.setMode(f.bs[1], Prefill, f.clk.Now().Add(time.Second))
	for _, b := range f.bs {
		readings[b.URL] = probeReading{kv: 0.99, waiting: 4}
	}
	f.m.SampleDecode(context.Background(), f.bs)
	f.m.Tick(f.bs)
	if got := count(f.modes(), Prefill); got != 2 {
		t.Errorf("modes %v: acted on an empty window", f.modes())
	}
}
