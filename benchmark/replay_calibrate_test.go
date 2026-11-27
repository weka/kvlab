package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/weka/kvlab/llm"
)

// fakeTokenizer serves chat completions (or messages) and reports a prompt
// of fixed + content/charsPerToken tokens, content being every string in the
// body but a thinking signature and a few structural keys. It remembers each
// body's content stamp so a test can tell probes from run requests.
type fakeTokenizer struct {
	charsPerToken float64
	fixed         int
	constant      int           // > 0: report this whatever the body (a server calibration cannot move)
	probeDelay    time.Duration // each calibration probe is answered this late
	lastProbe     atomic.Int64  // UnixNano at which the last probe was answered
	mu            sync.Mutex
	bodies        []string
	probes        atomic.Int32
}

func countContent(v any, key string) int {
	switch x := v.(type) {
	case string:
		switch key {
		case "signature", "model", "role", "type":
			return 0
		}
		return len(x)
	case []any:
		n := 0
		for _, e := range x {
			n += countContent(e, key)
		}
		return n
	case map[string]any:
		n := 0
		for k, e := range x {
			n += countContent(e, k)
		}
		return n
	}
	return 0
}

func (f *fakeTokenizer) server(t *testing.T, anthropic bool) *httptest.Server {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body any
		_ = json.Unmarshal(raw, &body)
		f.mu.Lock()
		f.bodies = append(f.bodies, string(raw))
		f.mu.Unlock()
		if strings.Contains(string(raw), "calib-") {
			f.probes.Add(1)
			if f.probeDelay > 0 {
				select {
				case <-time.After(f.probeDelay):
				case <-r.Context().Done():
					return
				}
			}
			defer func() { f.lastProbe.Store(time.Now().UnixNano()) }()
		}
		n := f.fixed + int(float64(countContent(body, ""))/f.charsPerToken)
		if f.constant > 0 {
			n = f.constant
		}
		w.Header().Set("Content-Type", "application/json")
		if anthropic {
			fmt.Fprintf(w, `{"content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":%d,"output_tokens":1}}`, n)
			return
		}
		fmt.Fprintf(w, `{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":%d,"completion_tokens":1,"total_tokens":%d}}`, n, n+1)
	}))
	t.Cleanup(ts.Close)
	return ts
}

// calibrationSession is one instance of agentic turns; the second is the
// first over 10k captured prompt tokens (captured at 3 bytes per token).
func calibrationSession() RouterReplaySession {
	sys := []RouterReplaySystemBlock{{Hash: "sys", Bytes: 6000}}
	tools := &RouterReplayToolsSpec{Count: 6, Bytes: 9000, Hash: "tools"}
	var msgs []RouterReplayMessage
	var reqs []RouterReplayRequest
	add := func(m ...RouterReplayMessage) {
		msgs = append(msgs, m...)
		req := RouterReplayRequest{RequestID: uint64(len(reqs) + 1), Ts: "2026-05-12T08:50:06Z", OutputTokens: 10,
			SystemBlocks: sys, Tools: tools, Messages: append([]RouterReplayMessage(nil), msgs...)}
		req.InputTokens = req.capturedInputBytes() / 3
		reqs = append(reqs, req)
	}
	add(RouterReplayMessage{Role: "user", Hash: "u1", BlockTypes: []string{"text"}, Bytes: 3000})
	add(RouterReplayMessage{Role: "assistant", Hash: "a1", BlockTypes: []string{"thinking", "tool_use"}, Bytes: 4000, ToolUseIDs: []string{"call_1"}},
		RouterReplayMessage{Role: "user", Hash: "r1", BlockTypes: []string{"tool_result"}, Bytes: 9000, ToolResultIDs: []string{"call_1"}})
	add(RouterReplayMessage{Role: "assistant", Hash: "a2", BlockTypes: []string{"text", "tool_use"}, Bytes: 2500, ToolUseIDs: []string{"call_2"}},
		RouterReplayMessage{Role: "user", Hash: "r2", BlockTypes: []string{"tool_result"}, Bytes: 12000, ToolResultIDs: []string{"call_2"}})
	add(RouterReplayMessage{Role: "user", Hash: "u2", BlockTypes: []string{"text"}, Bytes: 1500})
	return RouterReplaySession{SessionID: "cal", StartTs: "2026-05-12T08:50:06Z",
		Instances: []RouterReplayInstance{{InstanceID: "i1", Role: "main", Requests: reqs}}}
}

func writeCalibrationReplay(t *testing.T, sessions ...RouterReplaySession) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "replay.jsonl")
	var b strings.Builder
	b.WriteString(`{"_schema":"replay-v3","summary":{}}` + "\n")
	for _, s := range sessions {
		line, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(line)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

var calibrationDocs = strings.Repeat("The quick brown fox jumps over the lazy dog near the riverbank. ", 3000)

func calibrationCfg(t *testing.T, ts *httptest.Server, typ string) AutoBenchmarkConfig {
	return AutoBenchmarkConfig{
		Model:            fmt.Sprintf("dynamic/%s,type=%s,model=test-model", ts.URL, typ),
		RouterReplayFile: writeCalibrationReplay(t, calibrationSession()),
		ReplayNoStamp:    true,
	}
}

func calibrationPoster(t *testing.T, cfg AutoBenchmarkConfig) *replayPoster {
	t.Helper()
	p, err := newReplayPoster(cfg.Model, llm.APIKeys{}, "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// Calibration converges within 3% against a server that tokenizes the
// synthesized text far below the capture's density, on both wire formats,
// and picks the first request over 10k captured tokens as its probe.
func TestCalibrationConverges(t *testing.T) {
	for _, typ := range []string{"openai", "anthropic"} {
		f := &fakeTokenizer{charsPerToken: 4.6, fixed: 60}
		cfg := calibrationCfg(t, f.server(t, typ == "anthropic"), typ)
		res, err := resolveReplayScale(context.Background(), cfg, calibrationPoster(t, cfg), calibrationDocs)
		if err != nil {
			t.Fatalf("%s: %v (%s)", typ, err, res.describe())
		}
		if res.Source != scaleSourceCalibrated || res.Probe.turn != 2 || res.Probe.target <= calibrateMinPromptTokens {
			t.Fatalf("%s: probe turn %d source %s, want turn 2 (the first over 10k)", typ, res.Probe.turn, res.Source)
		}
		last := res.Steps[len(res.Steps)-1]
		if d := float64(last.ServerTokens-res.Probe.target) / float64(res.Probe.target); d > 0.03 || d < -0.03 {
			t.Fatalf("%s: accepted %d server tokens for %d captured", typ, last.ServerTokens, res.Probe.target)
		}
		if first := res.Steps[0]; first.Scale != 1 || float64(first.ServerTokens) > 0.8*float64(res.Probe.target) {
			t.Fatalf("%s: the default-sized probe should under-tokenize here: %+v", typ, first)
		}
		if res.Scale < 1.2 || len(res.Steps) > calibrateMaxProbes || int(f.probes.Load()) != len(res.Steps) {
			t.Fatalf("%s: scale %.4f after %d steps, %d probes: %s", typ, res.Scale, len(res.Steps), f.probes.Load(), res.describe())
		}
		t.Logf("%s: %s", typ, res.describe())
	}
}

// An explicit scale (or chars-per-token) skips calibration: nothing is sent.
func TestCalibrationExplicitBypass(t *testing.T) {
	f := &fakeTokenizer{charsPerToken: 4.6}
	cfg := calibrationCfg(t, f.server(t, false), "openai")
	cfg.ReplayTokenScale = 1.3125
	res, err := resolveReplayScale(context.Background(), cfg, calibrationPoster(t, cfg), calibrationDocs)
	if err != nil || res.Scale != 1.3125 || res.Source != scaleSourceGiven || f.probes.Load() != 0 {
		t.Fatalf("explicit scale: %+v %v, %d probes", res, err, f.probes.Load())
	}
	cfg.ReplayTokenScale, cfg.ReplayCharsPerToken = 0, 3.4
	res, err = resolveReplayScale(context.Background(), cfg, nil, calibrationDocs)
	if err != nil || res.Scale != 0 || res.Source != scaleSourceCharsPerToken || f.probes.Load() != 0 {
		t.Fatalf("chars-per-token: %+v %v", res, err)
	}
	if got := sizingLine(42, replaySizingResolution{Scale: 1.3125, Source: scaleSourceGiven}); got != "seed=42 replay-token-scale=1.3125 (given)" {
		t.Errorf("sizing line %q", got)
	}
}

// A server whose count does not move with the content cannot be calibrated:
// the run gets an error, not a guess.
func TestCalibrationDoesNotConverge(t *testing.T) {
	f := &fakeTokenizer{constant: 1234}
	cfg := calibrationCfg(t, f.server(t, false), "openai")
	_, err := resolveReplayScale(context.Background(), cfg, calibrationPoster(t, cfg), calibrationDocs)
	if err == nil || !strings.Contains(err.Error(), "did not come within 3%") {
		t.Fatalf("want a non-convergence error, got %v", err)
	}
	if n := f.probes.Load(); n == 0 || n > calibrateMaxProbes {
		t.Errorf("%d probes, want 1..%d", n, calibrateMaxProbes)
	}
}

// After calibration the run's requests carry the captured token volume
// (input volume conformity ~100% against the same tokenizer), and the probes
// are nowhere in the run's results: not in the request data, not in the
// completed count, not in the volume sums.
func TestCalibratedRunHitsCapturedVolumeAndExcludesProbes(t *testing.T) {
	f := &fakeTokenizer{charsPerToken: 4.6, fixed: 60}
	ts := f.server(t, false)
	cfg := calibrationCfg(t, ts, "openai")
	res, err := resolveReplayScale(context.Background(), cfg, calibrationPoster(t, cfg), calibrationDocs)
	if err != nil {
		t.Fatal(err)
	}
	probes := int(f.probes.Load())
	cfg.replayScale = res

	dir := t.TempDir()
	rdw, err := newRequestDataWriter(dir, "m", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	st := newTestAutoState(4)
	sess := calibrationSession()
	runRouterReplaySession(context.Background(), cfg, st, rdw, sess, 1,
		endpointPicker{}, 30*time.Second, calibrationDocs, newConcurrencyGate(4, false))
	_ = rdw.close()

	n := len(sess.Instances[0].Requests)
	if got := st.totalCompleted.Load(); got != int64(n) {
		t.Fatalf("completed %d, want the session's %d requests only", got, n)
	}
	if got := int(f.probes.Load()); got != probes {
		t.Fatalf("the run sent %d more calibration-stamped requests", got-probes)
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	c, err := readJSONLContents(files[0])
	if err != nil || len(c.records) != n {
		t.Fatalf("request data: %d rows (%v), want %d", len(c.records), err, n)
	}
	v := st.volume.load()
	if pct := float64(v.InServerTokens) / float64(v.InCapturedTokens); pct < 0.93 || pct > 1.07 {
		t.Fatalf("input volume conformity %.1f%% after calibration, want ~100%% (%s)", 100*pct, v.summary())
	}
	// The scale moves only what is tokenized: the probe-sized bytes ratio is
	// above 100% by about the scale on content.
	if v.InWireBytes <= v.InCapturedBytes {
		t.Errorf("wire bytes %d not above captured %d at scale %.4f", v.InWireBytes, v.InCapturedBytes, res.Scale)
	}
	t.Logf("scale %.4f: %s", res.Scale, v.summary())
}

// The scale multiplies tokenizable bytes only: a thinking signature and the
// message structure keep their captured sizes.
func TestScaleLeavesSignaturesAndStructure(t *testing.T) {
	m := RouterReplayMessage{Role: "assistant", Hash: "a", BlockTypes: []string{"thinking", "tool_use"}, Bytes: 4000, ToolUseIDs: []string{"c"}}
	plain := buildMessageContent(m, calibrationDocs, replaySizing{}, nil)
	scaled := buildMessageContent(m, calibrationDocs, replaySizing{Scale: 1.5}, nil)
	if plain[0]["signature"] != scaled[0]["signature"] {
		t.Error("the signature changed with the scale")
	}
	for _, c := range []struct {
		name   string
		get    func([]map[string]interface{}) string
		factor float64
	}{
		{"thinking", func(b []map[string]interface{}) string { return b[0]["thinking"].(string) }, 1.5},
		{"tool_use input", func(b []map[string]interface{}) string {
			return b[1]["input"].(map[string]interface{})["_padding"].(string)
		}, 1.5},
	} {
		a, b := len(c.get(plain)), len(c.get(scaled))
		if d := float64(b) - c.factor*float64(a); d > 1 || d < -1 {
			t.Errorf("%s: %d -> %d bytes at scale 1.5", c.name, a, b)
		}
	}
	spec := &RouterReplayToolsSpec{Count: 3, Bytes: 3000, Hash: "t"}
	if a, b := len(toolDescriptions(spec, calibrationDocs, replaySizing{})[0]), len(toolDescriptions(spec, calibrationDocs, replaySizing{Scale: 2})[0]); b != 2*a {
		t.Errorf("tool description %d -> %d at scale 2", a, b)
	}
}

// calibrationRunCfg is a whole router-replay run against ts: one slot, the
// calibration corpus, request data saved to dir.
func calibrationRunCfg(t *testing.T, ts *httptest.Server, dir string, timeout time.Duration) AutoBenchmarkConfig {
	cfg := calibrationCfg(t, ts, "openai")
	cfg.DocsContent = calibrationDocs
	cfg.SaveRequestDataDir = dir
	cfg.Timeout = timeout
	cfg.RequestTimeout = 10 * time.Second
	cfg.Concurrency, cfg.StartSeries, cfg.MaxSeries = 1, 1, 1
	return cfg
}

// --timeout is the duration of the load, counted from after calibration:
// probes slower than the whole --timeout still calibrate, and the run then
// starts and sends its requests (the one-pass corpus drains well inside the
// --timeout). Before, calibration ran under the
// --timeout deadline, so the probes ate the run's duration (here: all of it,
// and the run never started).
func TestTimeoutCountsFromAfterCalibration(t *testing.T) {
	f := &fakeTokenizer{charsPerToken: 4.6, fixed: 60, probeDelay: time.Second}
	ts := f.server(t, false)
	dir := t.TempDir()
	const timeout = 1500 * time.Millisecond
	cfg := calibrationRunCfg(t, ts, dir, timeout)

	if err := RunAutoBenchmark(context.Background(), cfg); err != nil {
		t.Fatalf("run: %v", err)
	}
	probes := int(f.probes.Load())
	calib := time.Duration(probes) * f.probeDelay
	if probes < 2 || calib <= timeout {
		t.Fatalf("premise: %d probes (%s) must outlast --timeout %s", probes, calib, timeout)
	}

	files, _ := filepath.Glob(filepath.Join(dir, "*", "*.jsonl"))
	if len(files) != 1 {
		t.Fatalf("request data files: %v", files)
	}
	c, err := readJSONLContents(files[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(c.records) == 0 {
		t.Fatal("no run requests: calibration consumed the --timeout")
	}
	// The run's time-zero (run_params ts) is after the last probe answered,
	// and every request was sent after it.
	if !c.hasParams {
		t.Fatal("no run_params header")
	}
	last := time.Unix(0, f.lastProbe.Load())
	if c.params.WrittenAt.Before(last) {
		t.Errorf("run params ts %s precedes the last probe %s", c.params.WrittenAt, last)
	}
	if got := time.Duration(c.params.CalibrationSec * float64(time.Second)); got < calib {
		t.Errorf("calibration_sec %.2f, want >= %s", c.params.CalibrationSec, calib)
	}
}

// Calibration has its own bound and fails the run clearly when it is
// exceeded, rather than hanging on a slow server.
func TestCalibrationTotalTimeoutFailsClearly(t *testing.T) {
	old := calibrateTotalTimeout
	calibrateTotalTimeout = 300 * time.Millisecond
	t.Cleanup(func() { calibrateTotalTimeout = old })
	f := &fakeTokenizer{charsPerToken: 4.6, fixed: 60, probeDelay: 5 * time.Second}
	cfg := calibrationRunCfg(t, f.server(t, false), t.TempDir(), time.Hour)

	start := time.Now()
	err := RunAutoBenchmark(context.Background(), cfg)
	if err == nil || !strings.Contains(err.Error(), "calibration exceeded its 300ms bound") {
		t.Fatalf("want a calibration-bound error, got %v", err)
	}
	if d := time.Since(start); d > 3*time.Second {
		t.Errorf("took %s to fail; the bound is 300ms", d)
	}
}
