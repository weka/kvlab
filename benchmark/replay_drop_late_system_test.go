package benchmark

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func msgHashes(msgs []RouterReplayMessage) string {
	var h []string
	for _, m := range msgs {
		h = append(h, m.Hash)
	}
	return strings.Join(h, ",")
}

func TestDropLateSystemLateDroppedLeadingKept(t *testing.T) {
	req := RouterReplayRequest{Messages: []RouterReplayMessage{
		{Role: "system", Hash: "s0"},
		{Role: "developer", Hash: "d0"},
		{Role: "user", Hash: "u1"},
		{Role: "assistant", Hash: "a1"},
		{Role: "system", Hash: "rem1"},
		{Role: "user", Hash: "u2"},
		{Role: "developer", Hash: "rem2"},
		{Role: "system", Hash: "rem3"},
	}}
	got, n := dropLateSystemMessages(req)
	if n != 3 {
		t.Fatalf("dropped %d, want 3", n)
	}
	if h := msgHashes(got.Messages); h != "s0,d0,u1,a1,u2" {
		t.Fatalf("kept %s, want s0,d0,u1,a1,u2", h)
	}
	if len(req.Messages) != 8 {
		t.Fatal("dropLateSystemMessages mutated its input")
	}
}

func TestDropLateSystemNoOp(t *testing.T) {
	for _, msgs := range [][]RouterReplayMessage{
		nil,
		{{Role: "system", Hash: "s0"}, {Role: "system", Hash: "s1"}},
		{{Role: "system", Hash: "s0"}, {Role: "user", Hash: "u1"}, {Role: "assistant", Hash: "a1"}},
	} {
		req := lateSystemRequest()
		req.Messages = msgs
		got, n := dropLateSystemMessages(req)
		if n != 0 || msgHashes(got.Messages) != msgHashes(msgs) {
			t.Fatalf("not a no-op for %s: dropped %d", msgHashes(msgs), n)
		}
		a, ca, _ := buildOpenAIChatCompletionsBody(req, sysFirstDocs, "m", "run", 0, 0, false, replaySizing{}, nil, "", "", false)
		b, cb, _ := buildOpenAIChatCompletionsBody(got, sysFirstDocs, "m", "run", 0, 0, false, replaySizing{}, nil, "", "", false)
		if !bytes.Equal(a, b) || ca != cb {
			t.Fatal("body changed for a request with no late system message")
		}
	}
}

// A session's turn N carries turn N-1's messages as its own leading messages.
// After the drop, every turn's message list (and its wire messages up to the
// previous turn's last one) must still extend the previous turn's.
func TestDropLateSystemSessionPrefixConsistent(t *testing.T) {
	full := []RouterReplayMessage{
		{Role: "system", Hash: "s0", BlockTypes: []string{"text"}, Bytes: 300},
		{Role: "user", Hash: "u1", BlockTypes: []string{"text"}, Bytes: 120},
		{Role: "assistant", Hash: "a1", BlockTypes: []string{"text", "tool_use"}, Bytes: 160, ToolUseIDs: []string{"c1"}},
		{Role: "user", Hash: "u2", BlockTypes: []string{"tool_result"}, Bytes: 120, ToolResultIDs: []string{"c1"}},
		{Role: "system", Hash: "rem1", BlockTypes: []string{"text"}, Bytes: 90},
		{Role: "assistant", Hash: "a2", BlockTypes: []string{"text"}, Bytes: 100},
		{Role: "user", Hash: "u3", BlockTypes: []string{"text"}, Bytes: 100},
		{Role: "system", Hash: "rem2", BlockTypes: []string{"text"}, Bytes: 90},
		{Role: "assistant", Hash: "a3", BlockTypes: []string{"text"}, Bytes: 100},
		{Role: "user", Hash: "u4", BlockTypes: []string{"text"}, Bytes: 100},
	}
	// Turn boundaries: each request ends at a user/tool or reminder message.
	ends := []int{2, 4, 5, 7, 8, 10}
	var prevMsgs []RouterReplayMessage
	var prevWire []interface{}
	for _, e := range ends {
		req := RouterReplayRequest{MaxTokens: 64, OutputTokens: 32, Messages: append([]RouterReplayMessage(nil), full[:e]...)}
		got, _ := dropLateSystemMessages(req)
		for _, m := range got.Messages[1:] {
			if isSystemRole(m.Role) {
				t.Fatalf("turn ending %d still sends late system %s", e, m.Hash)
			}
		}
		if !strings.HasPrefix(msgHashes(got.Messages), msgHashes(prevMsgs)) {
			t.Fatalf("turn ending %d: %s does not extend %s", e, msgHashes(got.Messages), msgHashes(prevMsgs))
		}
		body, _, err := buildOpenAIChatCompletionsBody(got, sysFirstDocs, "m", "run", 0, 0, false, replaySizing{}, nil, "", "", false)
		if err != nil {
			t.Fatal(err)
		}
		var parsed struct {
			Messages []interface{} `json:"messages"`
		}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatal(err)
		}
		// The previous turn's last wire message carries the tail ask, so
		// compare everything before it.
		if len(prevWire) > 0 {
			k := len(prevWire) - 1
			pa, _ := json.Marshal(prevWire[:k])
			pb, _ := json.Marshal(parsed.Messages[:k])
			if !bytes.Equal(pa, pb) {
				t.Fatalf("turn ending %d: wire messages do not extend the previous turn's", e)
			}
		}
		prevMsgs, prevWire = got.Messages, parsed.Messages
	}
}

func TestCountLateSystemMessages(t *testing.T) {
	sess := func(reqs ...[]string) string {
		type m struct {
			Role string `json:"role"`
		}
		type r struct {
			Messages []m `json:"messages"`
		}
		var rs []r
		for _, roles := range reqs {
			var ms []m
			for _, role := range roles {
				ms = append(ms, m{role})
			}
			rs = append(rs, r{ms})
		}
		b, _ := json.Marshal(map[string]interface{}{"instances": []interface{}{map[string]interface{}{"requests": rs}}})
		return string(b)
	}
	content := `{"_schema":"replay-v5"}` + "\n" +
		sess([]string{"system", "user"}, []string{"system", "user", "assistant", "system", "user", "system"}) + "\n" +
		sess([]string{"user", "developer"}) + "\n"
	path := filepath.Join(t.TempDir(), "r.jsonl")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	msgs, reqs, total, err := countLateSystemMessages(path)
	if err != nil || msgs != 3 || reqs != 2 || total != 3 {
		t.Fatalf("got msgs=%d reqs=%d total=%d err=%v, want 3/2/3", msgs, reqs, total, err)
	}
	if want := "Replay: --replay-drop-late-system drops 3 late system/developer messages from 2 of 3 requests (leading system messages kept)"; lateSystemStartupLine(msgs, reqs, total) != want {
		t.Fatalf("startup line = %q", lateSystemStartupLine(msgs, reqs, total))
	}
}
