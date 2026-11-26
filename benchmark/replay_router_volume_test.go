package benchmark

import (
	"encoding/json"
	"strings"
	"testing"
)

// volumeDocs needs no JSON escaping, so a marshaled length is content plus
// structure and nothing else.
var volumeDocs = strings.Repeat("lorem ipsum dolor sit amet ", 2000)

// anthropicMessages builds req as an Anthropic body and returns its messages
// as raw JSON, one per message.
func anthropicMessages(t *testing.T, req RouterReplayRequest) []json.RawMessage {
	t.Helper()
	body, _, err := buildAnthropicMessagesBody(req, volumeDocs, "m", "", 0, 0, false, replaySizing{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var parsed struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatal(err)
	}
	return parsed.Messages
}

// canonicalLen is the capture's measure of a message: json.Marshal of the
// decoded object.
func canonicalLen(t *testing.T, raw json.RawMessage) int {
	t.Helper()
	var m map[string]interface{}
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(m)
	return len(b)
}

func contentBytes(blocks []map[string]interface{}) int {
	n := 0
	for _, b := range blocks {
		for _, k := range []string{"text", "content", "thinking", "signature"} {
			if v, ok := b[k].(string); ok {
				n += len(v)
			}
		}
		if in, ok := b["input"].(map[string]interface{}); ok {
			n += len(in["_padding"].(string))
		}
	}
	return n
}

// A tool_result-only user message carries its whole captured size: the
// message, marshaled as the capture measured it, is exactly Bytes long.
func TestTextlessToolResultGetsFullBudget(t *testing.T) {
	m := RouterReplayMessage{Role: "user", Hash: "tr", BlockTypes: []string{"tool_result"},
		Bytes: 5000, ToolResultIDs: []string{"toolu_01ABCDEFGHIJKLMNOPQRSTUV"}}
	blocks := buildMessageContent(m, volumeDocs, replaySizing{}, nil)
	if got, want := contentBytes(blocks), 5000-messageSkeletonBytes(m, messageSkeleton(m)); got != want {
		t.Fatalf("tool_result content = %d bytes, want %d", got, want)
	}
	if got := len(blocks[0]["content"].(string)); got < 4850 {
		t.Fatalf("tool_result content = %d bytes, want ~5000", got)
	}
	req := RouterReplayRequest{MaxTokens: 8, Messages: []RouterReplayMessage{
		{Role: "assistant", Hash: "a", BlockTypes: []string{"tool_use"}, Bytes: 300, ToolUseIDs: []string{"toolu_01ABCDEFGHIJKLMNOPQRSTUV"}},
		m,
	}}
	msgs := anthropicMessages(t, req)
	for i, want := range []int{300, 5000} {
		if got := canonicalLen(t, msgs[i]); got != want {
			t.Errorf("message %d marshals to %d bytes, captured %d", i, got, want)
		}
	}
}

// A mixed message splits its budget: the content of all blocks together is
// the message's size less its structure, never one full budget per block.
func TestMixedMessageDoesNotOvershoot(t *testing.T) {
	m := RouterReplayMessage{Role: "user", Hash: "mix", BlockTypes: []string{"text", "tool_result", "tool_result"},
		Bytes: 9000, ToolResultIDs: []string{"toolu_a", "toolu_b"}, CacheControl: "ephemeral"}
	blocks := buildMessageContent(m, volumeDocs, replaySizing{}, nil)
	skel := messageSkeletonBytes(m, messageSkeleton(m))
	if got, want := contentBytes(blocks), 9000-skel; got != want {
		t.Fatalf("content = %d bytes, want %d (Bytes %d less skeleton %d)", got, want, 9000, skel)
	}
	if skel > 250 {
		t.Fatalf("skeleton %d bytes: structure should be small", skel)
	}
	// Split by type weight: each tool_result gets 2.364x the text.
	a, b, c := len(blocks[0]["text"].(string)), len(blocks[1]["content"].(string)), len(blocks[2]["content"].(string))
	if b-c > 1 || c-b > 1 || float64(b)/float64(a) < 2.3 || float64(b)/float64(a) > 2.43 {
		t.Fatalf("shares %d/%d/%d: want text:tool_result:tool_result = 1000:2364:2364", a, b, c)
	}
}

// tool_use input is sized to its share: thinking + tool_use, no text block,
// marshals to the captured size in the Anthropic body, the tool_use input
// taking its type's weight and the thinking block's share going mostly to
// its signature.
func TestToolUseInputSizedToShare(t *testing.T) {
	m := RouterReplayMessage{Role: "assistant", Hash: "tu", BlockTypes: []string{"thinking", "tool_use"},
		Bytes: 4000, ToolUseIDs: []string{"toolu_x"}}
	req := RouterReplayRequest{MaxTokens: 8, Messages: []RouterReplayMessage{
		{Role: "user", Hash: "u", BlockTypes: []string{"text"}, Bytes: 500},
		m,
		{Role: "user", Hash: "r", BlockTypes: []string{"tool_result"}, Bytes: 700, ToolResultIDs: []string{"toolu_x"}},
	}}
	msgs := anthropicMessages(t, req)
	for i, want := range []int{500, 4000} {
		if got := canonicalLen(t, msgs[i]); got != want {
			t.Errorf("message %d marshals to %d bytes, captured %d", i, got, want)
		}
	}
	blocks := buildMessageContent(m, volumeDocs, replaySizing{}, nil)
	pad := blocks[1]["input"].(map[string]interface{})["_padding"].(string)
	think, sig := blocks[0]["thinking"].(string), blocks[0]["signature"].(string)
	budget := 4000 - messageSkeletonBytes(m, messageSkeleton(m))
	shares := weightedShares(budget, []int{blockWeight("thinking"), blockWeight("tool_use")})
	if len(pad) != shares[1] || len(think)+len(sig) != shares[0] {
		t.Fatalf("tool_use padding %d, thinking %d + signature %d; want %v", len(pad), len(think), len(sig), shares)
	}
	if len(pad) < 700 || len(pad) > 900 || len(think) > len(sig)/5 {
		t.Fatalf("tool_use padding %d (want ~21%% of %d), thinking %d vs signature %d", len(pad), budget, len(think), len(sig))
	}
}

// Images ride as text of their share, not a fixed placeholder.
func TestImageSizedToShare(t *testing.T) {
	m := RouterReplayMessage{Role: "user", Hash: "img", BlockTypes: []string{"image", "text"}, Bytes: 3000}
	blocks := buildMessageContent(m, volumeDocs, replaySizing{}, nil)
	if blocks[0]["type"] != "text" || len(blocks[0]["text"].(string)) < 1400 {
		t.Fatalf("image block = %v bytes, want ~1/2 of the message", len(blocks[0]["text"].(string)))
	}
}

// The tools array marshals to the captured tools bytes.
func TestToolsArrayMatchesCapturedBytes(t *testing.T) {
	spec := &RouterReplayToolsSpec{Count: 7, Bytes: 12345, Hash: "tools-h"}
	b, _ := json.Marshal(buildTools(spec, volumeDocs, replaySizing{}))
	if len(b) != 12345 {
		t.Fatalf("tools array %d bytes, captured %d", len(b), 12345)
	}
	oa := buildOpenAITools(spec, volumeDocs, replaySizing{})
	for i, tool := range buildTools(spec, volumeDocs, replaySizing{}) {
		if oa[i]["function"].(map[string]interface{})["description"] != tool["description"] {
			t.Fatalf("tool %d: OpenAI description differs from Anthropic", i)
		}
	}
}

// The OpenAI path carries the same content: a text-less tool_result reaches
// the wire as a role=tool message of its full share, and tool_use arguments
// carry the padding of their share.
func TestOpenAIPathCarriesFullContent(t *testing.T) {
	tu := RouterReplayMessage{Role: "assistant", Hash: "tu", BlockTypes: []string{"thinking", "tool_use"},
		Bytes: 4000, ToolUseIDs: []string{"call_1"}}
	tr := RouterReplayMessage{Role: "user", Hash: "tr", BlockTypes: []string{"tool_result"},
		Bytes: 6000, ToolResultIDs: []string{"call_1"}}
	req := RouterReplayRequest{MaxTokens: 8, Messages: []RouterReplayMessage{
		{Role: "user", Hash: "u", BlockTypes: []string{"text"}, Bytes: 200}, tu, tr}}
	for _, sglang := range []bool{false, true} {
		body, _, err := buildOpenAIChatCompletionsBody(req, volumeDocs, "m", "", 0, 0, false, replaySizing{}, nil, "", "", sglang)
		if err != nil {
			t.Fatal(err)
		}
		var parsed struct {
			Messages []struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolCalls []struct {
					Function struct {
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatal(err)
		}
		var toolContent, args, thinking int
		for _, m := range parsed.Messages {
			switch m.Role {
			case "tool":
				toolContent += len(m.Content)
			case "assistant":
				thinking += len(m.Content)
				for _, c := range m.ToolCalls {
					args += len(c.Function.Arguments)
				}
			}
		}
		wantTR := contentBytes(buildMessageContent(tr, volumeDocs, replaySizing{}, nil))
		if toolContent != wantTR || toolContent < 5900 {
			t.Errorf("sglang=%v: tool message %d bytes, want %d (~6000)", sglang, toolContent, wantTR)
		}
		tb := buildMessageContent(tu, volumeDocs, replaySizing{}, nil)
		wantArgs := len(`{"_padding":""}`) + len(tb[1]["input"].(map[string]interface{})["_padding"].(string))
		if args != wantArgs || thinking != len(tb[0]["thinking"].(string)) {
			t.Errorf("sglang=%v: tool_call arguments %d, thinking %d bytes: want %d and %d (the signature is not sent)",
				sglang, args, thinking, wantArgs, len(tb[0]["thinking"].(string)))
		}
	}
}
