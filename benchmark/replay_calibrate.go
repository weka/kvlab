package benchmark

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/weka/kvlab/config"
)

// Replay content is synthesized text sized to the captured bytes, and the
// serving tokenizer counts that text at its own density, not the original
// traffic's: a byte-faithful replay puts ~45-75% of the captured prompt
// tokens through the server, depending on the model. Calibration measures
// the density on the running server before the run starts and scales every
// tokenizable part to compensate (replaySizing.Scale):
//
//  1. Probe: the first request the run would send (role, no-usage and
//     --limit-context filters applied, --replay-drop-late-system trimmed)
//     whose captured prompt exceeds calibrateMinPromptTokens. Sent with
//     max_tokens 1, its own content stamp (a probe nonce, so its KV is
//     nobody else's prefix), and never recorded in results or stats.
//  2. Adjust: the server's prompt tokens against the captured prompt give
//     the next scale, by a multiplicative step and then by secant on the
//     last two probes (the prompt is a fixed part — template, stamp,
//     instruction, structure — plus the scaled content).
//  3. Accept a scale whose probe lands within calibrateTolerance of the
//     captured prompt. A run that cannot get there in calibrateMaxProbes
//     fails rather than running at a size it could not hold.
//
// The accepted scale is printed beside the seed: --seed reproduces content,
// the scale reproduces sizes, and identical bytes need both
// (--seed=S --replay-token-scale=X). An explicit --replay-token-scale or
// --replay-chars-per-token skips calibration.
const (
	calibrateMinPromptTokens = 10_000
	calibrateTolerance       = 0.03
	calibrateMaxProbes       = 5 // the default-sized probe plus four adjusted ones
	// calibrateScanSessions bounds how far into the dispatch order the probe
	// search reads; past it the largest candidate seen is used.
	calibrateScanSessions = 2000
	calibrateProbeTimeout = 5 * time.Minute
)

// calibrateTotalTimeout bounds one model's whole calibration (probe pick and
// every probe). Calibration runs before --timeout starts counting, so it
// needs a bound of its own: a server that answers probes but never
// converges is already bounded by calibrateMaxProbes, and this catches the
// one that is merely slow. A var so tests can shorten it.
var calibrateTotalTimeout = 20 * time.Minute

// Where a run's replay sizing came from, for the startup line, summary and
// run params.
const (
	scaleSourceCalibrated    = "calibrated"
	scaleSourceGiven         = "given"
	scaleSourceCharsPerToken = "chars-per-token"
	scaleSourceDryRun        = "dry-run"
)

// calibrationProbe is the request calibration sends and the prompt size it
// aims for.
type calibrationProbe struct {
	req RouterReplayRequest
	// target is the captured prompt tokens of what is sent: the request's
	// InputTokens, less the share of any messages --replay-drop-late-system
	// left out (pro rata by captured bytes).
	target  int
	session string
	turn    int
}

// calibrationStep is one probe: the scale it was built at and what the
// server counted.
type calibrationStep struct {
	Scale        float64
	ServerTokens int
}

// pickCalibrationProbe reads the replay in dispatch order and returns the
// first request the run would send whose captured prompt is over
// calibrateMinPromptTokens, or the largest one within the scan when none is.
// ok is false when the replay holds no request the run would send.
func pickCalibrationProbe(cfg AutoBenchmarkConfig) (calibrationProbe, bool, error) {
	stream, err := openRouterReplayStream(cfg.RouterReplayFile, routerReplayStreamOpts{
		ChanCap:               2,
		SessionLimit:          cfg.ReplaySeries,
		AllowedIndices:        cfg.RouterReplaySeriesIndices,
		MaxRequestsPerSession: cfg.ReplayMaxRequestsPerSession,
		Strategy:              cfg.ReplayStrategy,
		From:                  cfg.ReplayFrom,
	})
	if err != nil {
		return calibrationProbe{}, false, err
	}
	defer stream.Close()
	defer stream.Stop()
	includeRole := buildRoleFilter(cfg.RouterReplayRoles)
	var best calibrationProbe
	found := false
	ctx := context.Background()
	for n := 0; n < calibrateScanSessions; n++ {
		sess, _, ok := stream.Pull(ctx)
		if !ok {
			break
		}
		for _, inst := range sess.Instances {
			if !includeRole(inst.Role) {
				continue
			}
			for ti, req := range inst.Requests {
				if skipNoUsage(cfg, req) || exceedsLimitContext(req, cfg.LimitContext) || req.InputTokens <= 0 {
					continue
				}
				p := calibrationProbe{req: req, target: req.InputTokens, session: sess.SessionID, turn: ti + 1}
				if cfg.ReplayDropLateSystem {
					full := req.capturedInputBytes()
					trimmed, dropped := dropLateSystemMessages(req)
					if dropped > 0 && full > 0 {
						p.req = trimmed
						p.target = int(int64(req.InputTokens) * int64(trimmed.capturedInputBytes()) / int64(full))
					}
				}
				if p.target > calibrateMinPromptTokens {
					return p, true, nil
				}
				if !found || p.target > best.target {
					best, found = p, true
				}
			}
		}
	}
	return best, found, nil
}

// probeNonce is a fresh content stamp for calibration probes.
func probeNonce() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return "calib-" + hex.EncodeToString(b)
}

// probePromptTokens sends req built at scale as a calibration probe and
// returns the prompt tokens the server reported. The probe is not a run
// request: max_tokens 1, not streamed, stamped with nonce, and nothing about
// it reaches the run's state.
func (p *replayPoster) probePromptTokens(ctx context.Context, req RouterReplayRequest, docs, nonce string, scale float64) (int, error) {
	req.Stream = false
	req.OutputTokens, req.MaxTokens = 1, 1
	req.passStamp = nonce
	if p.systemFirst {
		req = hoistSystemMessages(req)
	}
	sz := replaySizing{Scale: scale}
	var body []byte
	var err error
	if p.apiType == "anthropic" {
		body, _, err = buildAnthropicMessagesBody(req, docs, p.model, nonce, 0, 0, false, sz, nil)
	} else {
		body, _, err = buildOpenAIChatCompletionsBody(req, docs, p.model, nonce, 0, 0, false, sz, nil,
			p.reasoningEffort, p.thinking, p.apiType == "openai_sglang")
	}
	if err != nil {
		return 0, fmt.Errorf("build probe: %w", err)
	}
	ctx, cancel := context.WithTimeout(ctx, calibrateProbeTimeout)
	defer cancel()
	attempts := p.endpointAttempts()
	for i, u := range attempts {
		resp, err := p.sendOnce(ctx, u, body, false, nonce)
		if err != nil {
			return 0, err
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		if resp.StatusCode == http.StatusNotFound && i+1 < len(attempts) {
			continue
		}
		if resp.StatusCode/100 != 2 {
			return 0, fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
		}
		p.epMu.Lock()
		if p.epResolved == "" {
			p.epResolved = u
		}
		p.epMu.Unlock()
		return promptTokensFromUsage(raw)
	}
	return 0, fmt.Errorf("no endpoint answered")
}

// promptTokensFromUsage reads the whole prompt a response's usage reports:
// OpenAI prompt_tokens (cached included), or Anthropic input_tokens plus
// cache read and creation.
func promptTokensFromUsage(raw []byte) (int, error) {
	var r struct {
		Usage struct {
			PromptTokens             int `json:"prompt_tokens"`
			InputTokens              int `json:"input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return 0, fmt.Errorf("probe response: %w", err)
	}
	if n := r.Usage.PromptTokens; n > 0 {
		return n, nil
	}
	if n := r.Usage.InputTokens + r.Usage.CacheReadInputTokens + r.Usage.CacheCreationInputTokens; n > 0 {
		return n, nil
	}
	return 0, fmt.Errorf("probe response reports no prompt tokens: %.200s", raw)
}

// calibrateScale runs the probe loop against measure (a server probe in a
// run, a fake in tests) and returns the accepted scale and every step taken.
func calibrateScale(target int, measure func(scale float64) (int, error)) (float64, []calibrationStep, error) {
	if target <= 0 {
		return 0, nil, fmt.Errorf("probe has no captured prompt tokens")
	}
	var steps []calibrationStep
	scale := 1.0
	for i := 0; i < calibrateMaxProbes; i++ {
		got, err := measure(scale)
		if err != nil {
			return 0, steps, fmt.Errorf("probe at scale %.4f: %w", scale, err)
		}
		steps = append(steps, calibrationStep{Scale: scale, ServerTokens: got})
		if math.Abs(float64(got-target)) <= calibrateTolerance*float64(target) {
			return scale, steps, nil
		}
		next := scale * float64(target) / float64(max(got, 1))
		if n := len(steps); n >= 2 {
			a, b := steps[n-2], steps[n-1]
			if b.ServerTokens != a.ServerTokens {
				if s := b.Scale + float64(target-b.ServerTokens)*(b.Scale-a.Scale)/float64(b.ServerTokens-a.ServerTokens); s > 0 {
					next = s
				}
			}
		}
		// Four decimals is what the run prints, so a scale passed back with
		// --replay-token-scale reproduces the calibrated bytes exactly.
		scale = math.Round(next*1e4) / 1e4
	}
	return 0, steps, fmt.Errorf("server prompt did not come within %.0f%% of the captured %d tokens in %d probes",
		100*calibrateTolerance, target, calibrateMaxProbes)
}

// replaySizingResolution is the scale a run uses and where it came from.
type replaySizingResolution struct {
	Scale  float64
	Source string
	Probe  calibrationProbe
	Steps  []calibrationStep
	// Elapsed is the wall time calibration took (0 when it did not run):
	// setup time, outside the run's --timeout.
	Elapsed time.Duration
}

// calibrateModels resolves every model's replay sizing, concurrently (each
// model probes its own server), before any model's run starts. The error is
// the first model whose calibration failed; each failure is also printed.
func calibrateModels(ctx context.Context, cfg AutoBenchmarkConfig, models []string, docs string) ([]replaySizingResolution, error) {
	// SIGINT during calibration stops it; the run's own handler is not
	// installed yet.
	ctx, stop := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	out := make([]replaySizingResolution, len(models))
	errs := make([]error, len(models))
	var wg sync.WaitGroup
	for i, m := range models {
		mcfg := cfg
		mcfg.Model = m
		wg.Add(1)
		go func() {
			defer wg.Done()
			out[i], errs[i] = calibrateModel(ctx, mcfg, docs)
		}()
	}
	wg.Wait()
	for i, m := range models {
		name := shortModelName(m)
		if errs[i] != nil {
			fmt.Fprintf(os.Stderr, "\n[calibrate][%s] FAILED: %v\n"+
				"[calibrate] the run is not started: its prompts would not carry the captured token volume. "+
				"Pass --replay-token-scale=X to size explicitly (1 = byte-faithful, uncalibrated).\n\n",
				name, errs[i])
			continue
		}
		r := out[i]
		fmt.Printf("Replay sizing [%s]: %s\n", name, sizingLine(cfg.Seed, r))
		if r.Source == scaleSourceCalibrated {
			fmt.Printf("  reproduce these exact bytes with --seed=%d --replay-token-scale=%.4f (--seed alone reproduces content, not sizes)\n",
				cfg.Seed, r.Scale)
		}
	}
	if err := errors.Join(errs...); err != nil {
		return nil, fmt.Errorf("replay calibration: %w", err)
	}
	return out, nil
}

// calibrateModel resolves one model's sizing under calibrateTotalTimeout and
// logs the probes and the time taken.
func calibrateModel(ctx context.Context, cfg AutoBenchmarkConfig, docs string) (replaySizingResolution, error) {
	name := shortModelName(cfg.Model)
	var poster *replayPoster
	if !cfg.DryRun && cfg.ReplayTokenScale <= 0 && cfg.ReplayCharsPerToken <= 0 {
		p, err := newReplayPoster(cfg.Model, config.GetAPIKeys(), "", "", false, 0, 0, 0, nil, nil)
		if err != nil {
			return replaySizingResolution{}, err
		}
		p.systemFirst = cfg.ReplaySystemFirst
		poster = p
		fmt.Fprintf(os.Stderr, "[calibrate][%s] sizing replay content to the server's tokenizer "+
			"(bound %s; not counted in --timeout)...\n", name, calibrateTotalTimeout)
	}
	start := time.Now()
	cctx, cancel := context.WithTimeout(ctx, calibrateTotalTimeout)
	defer cancel()
	res, err := resolveReplayScale(cctx, cfg, poster, docs)
	if poster != nil {
		res.Elapsed = time.Since(start)
	}
	if len(res.Steps) > 0 {
		fmt.Fprintf(os.Stderr, "[calibrate][%s] %s\n", name, res.describe())
	}
	if err != nil && ctx.Err() == nil && errors.Is(cctx.Err(), context.DeadlineExceeded) {
		return res, fmt.Errorf("calibration exceeded its %s bound after %d probe(s): %w",
			calibrateTotalTimeout, len(res.Steps), err)
	}
	if err == nil && poster != nil {
		fmt.Fprintf(os.Stderr, "[calibrate][%s] done in %s (setup, outside --timeout)\n",
			name, res.Elapsed.Round(100*time.Millisecond))
	}
	return res, err
}

// resolveReplayScale decides a run's content scale: the explicit one when
// given, none under --replay-chars-per-token or a dry run, and otherwise a
// calibrated one. The error is a calibration that could not converge, or a
// probe the server refused.
func resolveReplayScale(ctx context.Context, cfg AutoBenchmarkConfig, poster *replayPoster, docs string) (replaySizingResolution, error) {
	switch {
	case cfg.ReplayTokenScale > 0:
		return replaySizingResolution{Scale: cfg.ReplayTokenScale, Source: scaleSourceGiven}, nil
	case cfg.ReplayCharsPerToken > 0:
		return replaySizingResolution{Source: scaleSourceCharsPerToken}, nil
	case cfg.DryRun:
		return replaySizingResolution{Scale: 1, Source: scaleSourceDryRun}, nil
	}
	probe, ok, err := pickCalibrationProbe(cfg)
	if err != nil {
		return replaySizingResolution{}, fmt.Errorf("pick a calibration probe: %w", err)
	}
	if !ok {
		return replaySizingResolution{}, fmt.Errorf("the replay holds no request this run would send")
	}
	if probe.target <= calibrateMinPromptTokens {
		fmt.Fprintf(os.Stderr, "[calibrate] warning: no request over %d captured prompt tokens in the first %d sessions; "+
			"probing the largest (%d tokens), where the template's fixed tokens weigh more\n",
			calibrateMinPromptTokens, calibrateScanSessions, probe.target)
	}
	nonce := probeNonce()
	scale, steps, err := calibrateScale(probe.target, func(s float64) (int, error) {
		return poster.probePromptTokens(ctx, probe.req, docs, nonce, s)
	})
	res := replaySizingResolution{Scale: scale, Source: scaleSourceCalibrated, Probe: probe, Steps: steps}
	return res, err
}

// describe is the calibration's account for the log: the probe and each
// step.
func (r replaySizingResolution) describe() string {
	var b strings.Builder
	fmt.Fprintf(&b, "probe session %s turn %d, captured prompt %d tokens:", r.Probe.session, r.Probe.turn, r.Probe.target)
	for _, s := range r.Steps {
		fmt.Fprintf(&b, " scale %.4f -> %d (%.1f%%);", s.Scale, s.ServerTokens, 100*float64(s.ServerTokens)/float64(max(r.Probe.target, 1)))
	}
	return strings.TrimSuffix(b.String(), ";")
}

// sizingLine is the run's reproduction line: seed and scale together.
func sizingLine(seed int64, r replaySizingResolution) string {
	scale := "off"
	if r.Scale > 0 {
		scale = fmt.Sprintf("%.4f", r.Scale)
	}
	return fmt.Sprintf("seed=%d replay-token-scale=%s (%s)", seed, scale, r.Source)
}
