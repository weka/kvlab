package benchmark

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/weka/kvlab/tools"
)

func volMetrics(captured, uncached, cached, out, target, capB, wireB int) RequestMetrics {
	return RequestMetrics{
		UsageData: tools.ExecutionUsageData{
			InputTokens:  tools.TokenUsage{Count: uncached},
			CachedTokens: tools.TokenUsage{Count: cached},
			OutputTokens: tools.TokenUsage{Count: out},
		},
		OutputTarget:        target,
		CapturedInputTokens: captured,
		CapturedInputBytes:  capB,
		WireInputBytes:      wireB,
	}
}

// Input conformity is server prompt tokens (uncached + cached) over the
// captured prompt, summed; errors and skips do not count, and a request the
// server reported no usage for adds nothing to the input side.
func TestVolumeCountersSums(t *testing.T) {
	var c volumeCounters
	a := volMetrics(1000, 200, 600, 90, 100, 3000, 3030)
	b := volMetrics(3000, 1000, 1200, 180, 200, 9000, 9000)
	noUsage := volMetrics(500, 0, 0, 0, 50, 1500, 1500)
	failed := volMetrics(9999, 1, 1, 1, 9999, 9999, 9999)
	failed.Error = errors.New("status 400")
	skipped := volMetrics(9999, 1, 1, 1, 9999, 9999, 9999)
	skipped.Skipped = true
	for _, m := range []RequestMetrics{a, b, noUsage, failed, skipped} {
		c.observe(&m)
	}
	v := c.load()
	want := volumeTotals{InCapturedTokens: 4000, InServerTokens: 3000, InCapturedBytes: 12000, InWireBytes: 12030,
		OutTargetTokens: 350, OutActualTokens: 270}
	if v != want {
		t.Fatalf("totals = %+v, want %+v", v, want)
	}
	if got := v.summary(); got != "input 75.0% (bytes 100.2%), output 77.1%" {
		t.Fatalf("summary = %q", got)
	}
}

// The request-data row carries the operands under volume_* names and the
// per-request ratios; an errored request keeps its operands but no ratios.
func TestRequestDataVolumeFields(t *testing.T) {
	var rec requestDataRecord
	m := volMetrics(1000, 300, 500, 45, 50, 4000, 4100)
	rec.applyVolume(&m)
	b, _ := json.Marshal(rec)
	var got map[string]any
	_ = json.Unmarshal(b, &got)
	for k, v := range map[string]float64{
		"volume_captured_input_tokens": 1000, "volume_captured_input_bytes": 4000,
		"volume_wire_input_bytes": 4100, "volume_output_target_tokens": 50,
		"volume_input": 0.8, "volume_input_bytes": 1.025, "volume_output": 0.9,
	} {
		if got[k] != v {
			t.Errorf("%s = %v, want %v", k, got[k], v)
		}
	}

	var bad requestDataRecord
	m.Error = errors.New("boom")
	bad.applyVolume(&m)
	b, _ = json.Marshal(bad)
	if s := string(b); strings.Contains(s, `"volume_input"`) || !strings.Contains(s, `"volume_captured_input_tokens":1000`) {
		t.Fatalf("errored row: %s", s)
	}
}

// capturedInputBytes is the sum the wire is held to, and keeps the
// pre-trim figure once dispatch has stored it.
func TestCapturedInputBytes(t *testing.T) {
	req := RouterReplayRequest{
		SystemBlocks: []RouterReplaySystemBlock{{Bytes: 100}, {Bytes: 900}},
		Tools:        &RouterReplayToolsSpec{Count: 2, Bytes: 5000},
		Messages: []RouterReplayMessage{
			{Role: "user", Bytes: 300},
			{Role: "assistant", Bytes: 200},
			{Role: "system", Bytes: 50},
		},
	}
	if got := req.capturedInputBytes(); got != 6550 {
		t.Fatalf("capturedInputBytes = %d, want 6550", got)
	}
	req.capturedBytes = req.capturedInputBytes()
	trimmed, _ := dropLateSystemMessages(req)
	if got := trimmed.capturedInputBytes(); got != 6550 {
		t.Fatalf("after drop = %d, want the captured 6550", got)
	}
}

// The posted body is held to the captured bytes: an Anthropic body for a
// replay request lands within its fixed extras (stamp, instruction, length
// ask, model/max_tokens keys) of the capture; the OpenAI body differs by its
// lighter message structure, its heavier tool wrapper and the thinking
// signatures it cannot carry.
func TestWireBytesNearCaptured(t *testing.T) {
	req := RouterReplayRequest{
		MaxTokens: 8,
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "hdr", Bytes: 120},
			{Hash: "s1", Bytes: 4000, CacheControl: "ephemeral"}},
		Tools: &RouterReplayToolsSpec{Count: 12, Bytes: 30000, Hash: "t"},
		Messages: []RouterReplayMessage{
			{Role: "user", Hash: "u", BlockTypes: []string{"text"}, Bytes: 2000},
			{Role: "assistant", Hash: "a", BlockTypes: []string{"thinking", "tool_use"}, Bytes: 6000, ToolUseIDs: []string{"toolu_1"}},
			{Role: "user", Hash: "r", BlockTypes: []string{"tool_result"}, Bytes: 20000, ToolResultIDs: []string{"toolu_1"}},
		},
	}
	captured := req.capturedInputBytes()
	anth, _, _ := buildAnthropicMessagesBody(req, volumeDocs, "m", "run1", 0, 0, false, replaySizing{}, nil)
	oa, _, _ := buildOpenAIChatCompletionsBody(req, volumeDocs, "m", "run1", 0, 0, false, replaySizing{}, nil, "", "", false)
	// The <200B header system block is not sent (effectiveSystemBlocks).
	if d := len(anth) - (captured - 120); d < 0 || d > 600 {
		t.Errorf("anthropic body %d bytes vs captured %d (less 120B header): off by %d", len(anth), captured-120, d)
	}
	// OpenAI has no place for a thinking signature, so it is not sent.
	sig := len(buildMessageContent(req.Messages[1], volumeDocs, replaySizing{}, nil)[0]["signature"].(string))
	if d := len(oa) - (captured - 120 - sig); d > 1500 || d < -1500 {
		t.Errorf("openai body %d bytes vs captured %d: off by %d", len(oa), captured-120, d)
	}
}
