package benchmark

import (
	"fmt"
	"sync/atomic"
)

// Volume conformity: how much of the captured request volume a replay put
// through the server, input and output alike.
//
//   - input: prompt tokens the server reported (uncached + cached) over the
//     prompt tokens the capture recorded for the same requests (its
//     input_tokens, which is the whole prompt: prefill + cache read + cache
//     creation). Replay content is synthesized text, so this ratio carries
//     the serving tokenizer's density on it as well as the sizing; with the
//     content scale calibrated (replay_calibrate.go) it sits near 100%.
//   - input bytes: the request body as posted over the capture's request
//     bytes (system blocks + tools + messages, each the canonical JSON the
//     capture measured). Replay sizes every part to its captured bytes, so
//     this sits at ~100% at scale 1 and above it by about a calibrated
//     content scale: a little above from the run stamp, the output
//     instruction and the length ask, below where --replay-drop-late-system
//     leaves messages out or where the OpenAI shape spends fewer bytes on
//     structure than the capture's Anthropic one and drops thinking
//     signatures.
//   - output: completion tokens over the max_tokens each request carried.
//     ~100% under ignore_eos by construction; under --verify it scores how
//     well the per-request length ask alone reproduces the captured output.
//
// Each ratio sums over the requests that completed without error and carried
// its operands, so a run's figure is a token-weighted (or byte-weighted)
// ratio, not a mean of per-request ratios.

// volumeTotals is one snapshot of the run's volume sums.
type volumeTotals struct {
	InCapturedTokens int64 // capture's prompt tokens, requests with server usage
	InServerTokens   int64 // server-reported prompt tokens (uncached + cached), same requests
	InCapturedBytes  int64 // capture's request bytes, same requests
	InWireBytes      int64 // posted body bytes, same requests
	OutTargetTokens  int64 // max_tokens on the wire
	OutActualTokens  int64 // completion tokens returned
}

// volumeCounters accumulates volumeTotals across concurrent sessions.
type volumeCounters struct {
	inCapturedTokens atomic.Int64
	inServerTokens   atomic.Int64
	inCapturedBytes  atomic.Int64
	inWireBytes      atomic.Int64
	outTargetTokens  atomic.Int64
	outActualTokens  atomic.Int64
}

// observe adds one finished request.
func (c *volumeCounters) observe(m *RequestMetrics) {
	if m.Error != nil || m.Skipped {
		return
	}
	if server := m.UsageData.InputTokens.Count + m.UsageData.CachedTokens.Count; m.CapturedInputTokens > 0 && server > 0 {
		c.inCapturedTokens.Add(int64(m.CapturedInputTokens))
		c.inServerTokens.Add(int64(server))
		c.inCapturedBytes.Add(int64(m.CapturedInputBytes))
		c.inWireBytes.Add(int64(m.WireInputBytes))
	}
	if m.OutputTarget > 0 {
		c.outTargetTokens.Add(int64(m.OutputTarget))
		c.outActualTokens.Add(int64(m.UsageData.OutputTokens.Count))
	}
}

func (c *volumeCounters) load() volumeTotals {
	return volumeTotals{
		InCapturedTokens: c.inCapturedTokens.Load(),
		InServerTokens:   c.inServerTokens.Load(),
		InCapturedBytes:  c.inCapturedBytes.Load(),
		InWireBytes:      c.inWireBytes.Load(),
		OutTargetTokens:  c.outTargetTokens.Load(),
		OutActualTokens:  c.outActualTokens.Load(),
	}
}

// any reports whether there is anything to render: a replay run with at
// least one completed request.
func (v volumeTotals) any() bool { return v.InCapturedTokens > 0 || v.OutTargetTokens > 0 }

func volPct(num, den int64) string {
	if den <= 0 {
		return "-"
	}
	return fmt.Sprintf("%.1f%%", 100*float64(num)/float64(den))
}

// summary is the summary's one line: "input X% (bytes Y%), output Z%".
func (v volumeTotals) summary() string {
	return fmt.Sprintf("input %s (bytes %s), output %s",
		volPct(v.InServerTokens, v.InCapturedTokens), volPct(v.InWireBytes, v.InCapturedBytes),
		volPct(v.OutActualTokens, v.OutTargetTokens))
}

// detail backs the summary line with the sums it divides.
func (v volumeTotals) detail() string {
	return fmt.Sprintf("input %s of %s captured tokens, %s of %s captured bytes; output %s of %s requested",
		formatKiloInt(v.InServerTokens), formatKiloInt(v.InCapturedTokens),
		formatKiloInt(v.InWireBytes), formatKiloInt(v.InCapturedBytes),
		formatKiloInt(v.OutActualTokens), formatKiloInt(v.OutTargetTokens))
}

// progress is the progress line's segment, "" when there is nothing yet.
func (v volumeTotals) progress() string {
	if !v.any() {
		return ""
	}
	return fmt.Sprintf(" vol=in:%s,bytes:%s,out:%s",
		volPct(v.InServerTokens, v.InCapturedTokens), volPct(v.InWireBytes, v.InCapturedBytes),
		volPct(v.OutActualTokens, v.OutTargetTokens))
}

// volumeRatio is a per-request ratio for the request data, 0 when either
// operand is missing.
func volumeRatio(num, den int) float64 {
	if num <= 0 || den <= 0 {
		return 0
	}
	return round4(float64(num) / float64(den))
}

// applyVolume copies a request's volume operands into its request-data row
// and derives the per-request ratios.
func (rec *requestDataRecord) applyVolume(m *RequestMetrics) {
	rec.VolumeCapturedInputTokens = m.CapturedInputTokens
	rec.VolumeCapturedInputBytes = m.CapturedInputBytes
	rec.VolumeWireInputBytes = m.WireInputBytes
	rec.VolumeOutputTargetTokens = m.OutputTarget
	if m.Error != nil {
		return
	}
	rec.VolumeInput = volumeRatio(m.UsageData.InputTokens.Count+m.UsageData.CachedTokens.Count, m.CapturedInputTokens)
	if rec.VolumeInput > 0 {
		rec.VolumeInputBytes = volumeRatio(m.WireInputBytes, m.CapturedInputBytes)
	}
	if m.OutputTarget > 0 {
		rec.VolumeOutput = volumeRatio(m.UsageData.OutputTokens.Count, m.OutputTarget)
	}
}
