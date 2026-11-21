package benchmark

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

const sysFirstDocs = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" +
	"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" +
	"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" +
	"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

// lateSystemRequest is a captured agentic turn: a leading system block, then
// user / assistant tool_use / user tool_result, with a per-turn reminder as a
// system message after the tool turns and a developer message after that.
func lateSystemRequest() RouterReplayRequest {
	return RouterReplayRequest{
		MaxTokens:    64,
		OutputTokens: 32,
		SystemBlocks: []RouterReplaySystemBlock{{Hash: "sys-a", Bytes: 250}},
		Messages: []RouterReplayMessage{
			{Role: "user", Hash: "u1", BlockTypes: []string{"text"}, Bytes: 120},
			{Role: "assistant", Hash: "a1", BlockTypes: []string{"text", "tool_use"}, Bytes: 160, ToolUseIDs: []string{"call_1"}},
			{Role: "user", Hash: "u2", BlockTypes: []string{"tool_result"}, Bytes: 120, ToolResultIDs: []string{"call_1"}},
			{Role: "system", Hash: "rem1", BlockTypes: []string{"text"}, Bytes: 90},
			{Role: "user", Hash: "u3", BlockTypes: []string{"text"}, Bytes: 100},
			{Role: "developer", Hash: "dev1", BlockTypes: []string{"text"}, Bytes: 70},
		},
	}
}

func TestHoistSystemMessagesOrder(t *testing.T) {
	req := lateSystemRequest()
	got := hoistSystemMessages(req)
	var hashes, roles []string
	for _, m := range got.Messages {
		hashes = append(hashes, m.Hash)
		roles = append(roles, m.Role)
	}
	if want := "rem1,dev1,u1,a1,u2,u3"; strings.Join(hashes, ",") != want {
		t.Fatalf("order = %s, want %s", strings.Join(hashes, ","), want)
	}
	if want := "system,system,user,assistant,user,user"; strings.Join(roles, ",") != want {
		t.Fatalf("roles = %s, want %s", strings.Join(roles, ","), want)
	}
	if got.hoistedSystem != 2 {
		t.Fatalf("hoistedSystem = %d, want 2", got.hoistedSystem)
	}
	// The caller's request is untouched (posters share session data).
	if req.Messages[3].Hash != "rem1" || req.Messages[5].Role != "developer" {
		t.Fatal("hoistSystemMessages mutated its input")
	}
}

// openAIRoles returns the role sequence of an OpenAI chat body.
func openAIRoles(t *testing.T, body []byte) []string {
	t.Helper()
	var parsed struct {
		Messages []map[string]interface{} `json:"messages"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatal(err)
	}
	var roles []string
	for _, m := range parsed.Messages {
		roles = append(roles, m["role"].(string))
	}
	return roles
}

func TestSystemFirstOpenAINoLateSystemAfterConversation(t *testing.T) {
	req := hoistSystemMessages(lateSystemRequest())
	body, _, err := buildOpenAIChatCompletionsBody(req, sysFirstDocs, "m", "run", 0, 0, false, 0, nil, "", "", false)
	if err != nil {
		t.Fatal(err)
	}
	roles := openAIRoles(t, body)
	seenConv := false
	nSys := 0
	for _, r := range roles {
		if r == "system" {
			nSys++
			if seenConv {
				t.Fatalf("system message after a conversation turn: %v", roles)
			}
		} else {
			seenConv = true
		}
	}
	// RUN_GUID + 1 system block + output instruction + 2 hoisted.
	if nSys != 5 {
		t.Fatalf("system messages = %d, want 5 (%v)", nSys, roles)
	}

	// Without the flag the reminder stays where it was captured.
	plain, _, _ := buildOpenAIChatCompletionsBody(lateSystemRequest(), sysFirstDocs, "m", "run", 0, 0, false, 0, nil, "", "", false)
	if r := openAIRoles(t, plain); !strings.Contains(strings.Join(r, ","), "tool,system") {
		t.Fatalf("flag-off body lost the captured late system message: %v", r)
	}
}

func TestSystemFirstNoSystemMessagesUnchanged(t *testing.T) {
	req := lateSystemRequest()
	var conv []RouterReplayMessage
	for _, m := range req.Messages {
		if !isSystemRole(m.Role) {
			conv = append(conv, m)
		}
	}
	req.Messages = conv
	hoisted := hoistSystemMessages(req)
	if hoisted.hoistedSystem != 0 {
		t.Fatalf("hoistedSystem = %d, want 0", hoisted.hoistedSystem)
	}
	a, ca, _ := buildOpenAIChatCompletionsBody(req, sysFirstDocs, "m", "run", 0, 0, false, 0, nil, "", "", false)
	b, cb, _ := buildOpenAIChatCompletionsBody(hoisted, sysFirstDocs, "m", "run", 0, 0, false, 0, nil, "", "", false)
	if !bytes.Equal(a, b) || ca != cb {
		t.Fatal("OpenAI body changed for a request with no system messages")
	}
	a, ca, _ = buildAnthropicMessagesBody(req, sysFirstDocs, "m", "run", 0, 0, false, 0, nil)
	b, cb, _ = buildAnthropicMessagesBody(hoisted, sysFirstDocs, "m", "run", 0, 0, false, 0, nil)
	if !bytes.Equal(a, b) || ca != cb {
		t.Fatal("Anthropic body changed for a request with no system messages")
	}
}

func TestSystemFirstAnthropicSystemArray(t *testing.T) {
	req := hoistSystemMessages(lateSystemRequest())
	body, _, err := buildAnthropicMessagesBody(req, sysFirstDocs, "m", "run", 0, 0, false, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	var parsed struct {
		System   []map[string]interface{} `json:"system"`
		Messages []map[string]interface{} `json:"messages"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatal(err)
	}
	// RUN_GUID + 1 block + instruction + 2 hoisted.
	if len(parsed.System) != 5 {
		t.Fatalf("system blocks = %d, want 5", len(parsed.System))
	}
	if parsed.System[3]["text"] != hoistedSystemText(req.Messages[0], sysFirstDocs, 0) ||
		parsed.System[4]["text"] != hoistedSystemText(req.Messages[1], sysFirstDocs, 0) {
		t.Fatal("hoisted system text out of order")
	}
	for _, m := range parsed.Messages {
		if m["role"] == "system" {
			t.Fatal("system role left inside messages")
		}
	}
	if len(parsed.Messages) != 4 {
		t.Fatalf("messages = %d, want 4", len(parsed.Messages))
	}
}

// canonicalOrderIn asserts each needle occurs in hay, in the given order.
func canonicalOrderIn(t *testing.T, label, hay string, needles []string) {
	t.Helper()
	pos := 0
	for i, n := range needles {
		k := strings.Index(hay[pos:], n)
		if k < 0 {
			t.Fatalf("%s: piece %d missing or out of order", label, i)
		}
		pos += k + len(n)
	}
}

// The estimator's canonical text must follow the wire order: hoisted system
// text before the conversation (and, on Anthropic, before the tools).
func TestSystemFirstCanonicalMatchesWire(t *testing.T) {
	req := hoistSystemMessages(lateSystemRequest())
	req.Tools = &RouterReplayToolsSpec{Count: 1, Bytes: 400, Hash: "tools"}

	_, canon, _ := buildOpenAIChatCompletionsBody(req, sysFirstDocs, "m", "run", 0, 0, false, 0, nil, "", "", false)
	var pieces []string
	for _, m := range req.Messages {
		for _, blk := range buildMessageContent(m, sysFirstDocs, 0, nil) {
			for _, k := range []string{"text", "content"} {
				if v, ok := blk[k]; ok && v.(string) != "" {
					pieces = append(pieces, v.(string))
				}
			}
		}
	}
	canonicalOrderIn(t, "openai", canon, pieces)

	body, canon, _ := buildAnthropicMessagesBody(req, sysFirstDocs, "m", "run", 0, 0, false, 0, nil)
	var parsed struct {
		System []map[string]interface{} `json:"system"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatal(err)
	}
	toolText := synthText("tools:tool:0", 400-120, sysFirstDocs)
	wireOrder := []string{
		parsed.System[3]["text"].(string),
		parsed.System[4]["text"].(string),
		toolText,
	}
	for _, m := range req.Messages[2:] {
		for _, blk := range buildMessageContent(m, sysFirstDocs, 0, nil) {
			if v, ok := blk["text"]; ok {
				wireOrder = append(wireOrder, v.(string))
			}
		}
	}
	canonicalOrderIn(t, "anthropic", canon, wireOrder)
	// Same text as the captured-order canonical, only moved: nothing is
	// dropped or carried twice (every system message here is one block).
	plain := lateSystemRequest()
	plain.Tools = req.Tools
	_, plainCanon, _ := buildAnthropicMessagesBody(plain, sysFirstDocs, "m", "run", 0, 0, false, 0, nil)
	if len(canon) != len(plainCanon) {
		t.Fatalf("anthropic canonical length %d, captured order %d", len(canon), len(plainCanon))
	}
}
