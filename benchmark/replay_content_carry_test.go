package benchmark

import (
	"encoding/json"
	"strings"
	"testing"
)

// ccDelta is the capture's size of m's cache_control marker on its last
// block, for ttl "" or "1h".
func ccDelta(t *testing.T, m RouterReplayMessage, ttl string) int {
	t.Helper()
	cc := map[string]string{"type": "ephemeral"}
	if ttl != "" {
		cc["ttl"] = ttl
	}
	skel := messageSkeleton(m)
	bare, _ := json.Marshal(map[string]interface{}{"role": m.Role, "content": skel})
	skel[len(skel)-1]["cache_control"] = cc
	marked, _ := json.Marshal(map[string]interface{}{"role": m.Role, "content": skel})
	return len(marked) - len(bare)
}

// twoTurn is an instance whose first request ends in last (carrying
// cache_control) and whose second repeats it as next, then adds a turn.
func twoTurn(last, next RouterReplayMessage) RouterReplaySession {
	asst := RouterReplayMessage{Role: "assistant", Hash: "asst", BlockTypes: []string{"text"}, Bytes: 900}
	tail := RouterReplayMessage{Role: "user", Hash: "tail", BlockTypes: []string{"text"}, Bytes: 700, CacheControl: "ephemeral"}
	first := RouterReplayMessage{Role: "user", Hash: "first", BlockTypes: []string{"text"}, Bytes: 3000}
	return RouterReplaySession{SessionID: "s", Instances: []RouterReplayInstance{{Requests: []RouterReplayRequest{
		{Messages: []RouterReplayMessage{first, last}},
		{Messages: []RouterReplayMessage{first, next, asst, tail}},
	}}}}
}

func toolResultMsg(hash string, bytes int, cc string) RouterReplayMessage {
	return RouterReplayMessage{Role: "user", Hash: hash, BlockTypes: []string{"tool_result"},
		ToolResultIDs: []string{"toolu_1"}, Bytes: bytes, CacheControl: cc}
}

// sentMessage is message i of a built request body (after any leading
// system messages, which the OpenAI paths add), with cache_control
// removed, as JSON: the content the server tokenizes for it.
func sentMessage(t *testing.T, body []byte, i int) string {
	t.Helper()
	var b struct {
		Messages []map[string]interface{} `json:"messages"`
	}
	if err := json.Unmarshal(body, &b); err != nil {
		t.Fatal(err)
	}
	for len(b.Messages) > 0 && b.Messages[0]["role"] == "system" {
		b.Messages = b.Messages[1:]
	}
	m := b.Messages[i]
	if blocks, ok := m["content"].([]interface{}); ok {
		for _, blk := range blocks {
			if mb, ok := blk.(map[string]interface{}); ok {
				delete(mb, "cache_control")
			}
		}
	}
	out, _ := json.Marshal(m)
	return string(out)
}

// bodies builds req as the Anthropic, OpenAI and sglang paths send it.
func bodies(t *testing.T, req RouterReplayRequest, docs string, inj *uuidInjection) map[string][]byte {
	t.Helper()
	sz := replaySizing{Scale: 1.3}
	a, _, err := buildAnthropicMessagesBody(req, docs, "m", "", 0, 0, false, sz, inj)
	if err != nil {
		t.Fatal(err)
	}
	o, _, err := buildOpenAIChatCompletionsBody(req, docs, "m", "", 0, 0, false, sz, inj, "", "", false)
	if err != nil {
		t.Fatal(err)
	}
	s, _, err := buildOpenAIChatCompletionsBody(req, docs, "m", "", 0, 0, false, sz, inj, "", "", true)
	if err != nil {
		t.Fatal(err)
	}
	return map[string][]byte{"anthropic": a, "openai": o, "sglang": s}
}

func TestCarryContentSeedsCacheControlOnlyChange(t *testing.T) {
	docs := strings.Repeat("the quick brown fox jumps over the lazy dog. ", 400)
	base := toolResultMsg("x", 0, "")
	for _, ttl := range []string{"", "1h"} {
		d := ccDelta(t, base, ttl)
		last := toolResultMsg("with-cc", 5000+d, "ephemeral")
		next := toolResultMsg("without-cc", 5000, "")
		sess := twoTurn(last, next)
		carryContentSeeds(&sess)
		reqs := sess.Instances[0].Requests
		b1, b2 := bodies(t, reqs[0], docs, nil), bodies(t, reqs[1], docs, nil)
		for path := range b1 {
			if sentMessage(t, b1[path], 1) != sentMessage(t, b2[path], 1) {
				t.Errorf("ttl %q %s: message without cache_control sent with new content", ttl, path)
			}
		}
		// The marker itself is still sent as captured.
		if !strings.Contains(string(b1["anthropic"]), `"cache_control"`) {
			t.Errorf("ttl %q: first request lost its cache_control", ttl)
		}
		if got := reqs[1].Messages[1].contentSeed(); got != "with-cc" {
			t.Errorf("ttl %q: seed %q, want with-cc", ttl, got)
		}
	}
}

func TestCarryContentSeedsStringFormText(t *testing.T) {
	m := RouterReplayMessage{Role: "user", BlockTypes: []string{"text"}}
	skel, _ := json.Marshal(messageSkeleton(m))
	wrapper := len(skel) - len(`""`)
	d := ccDelta(t, m, "") + wrapper
	last := RouterReplayMessage{Role: "user", Hash: "a", BlockTypes: []string{"text"}, Bytes: 2000 + d, CacheControl: "ephemeral"}
	next := RouterReplayMessage{Role: "user", Hash: "b", BlockTypes: []string{"text"}, Bytes: 2000}
	if !cacheControlOnlyChange(last, next) {
		t.Fatalf("string-form drop of the marker (%d bytes) not recognized", d)
	}
}

func TestCarryContentSeedsRealChangeGetsNewContent(t *testing.T) {
	docs := strings.Repeat("lorem ipsum dolor sit amet ", 600)
	d := ccDelta(t, toolResultMsg("x", 0, ""), "")
	for name, next := range map[string]RouterReplayMessage{
		"bytes":   toolResultMsg("without-cc", 5000+d-100, ""),
		"tool id": {Role: "user", Hash: "without-cc", BlockTypes: []string{"tool_result"}, ToolResultIDs: []string{"toolu_2"}, Bytes: 5000},
		"blocks":  {Role: "user", Hash: "without-cc", BlockTypes: []string{"tool_result", "text"}, ToolResultIDs: []string{"toolu_1"}, Bytes: 5000},
	} {
		sess := twoTurn(toolResultMsg("with-cc", 5000+d, "ephemeral"), next)
		carryContentSeeds(&sess)
		reqs := sess.Instances[0].Requests
		if got := reqs[1].Messages[1].contentSeed(); got != "without-cc" {
			t.Errorf("%s: seed %q, want its own hash", name, got)
		}
		b1, b2 := bodies(t, reqs[0], docs, nil), bodies(t, reqs[1], docs, nil)
		if sentMessage(t, b1["openai"], 1) == sentMessage(t, b2["openai"], 1) {
			t.Errorf("%s: changed message sent with the old content", name)
		}
	}
}

// A carried seed sticks to the hash for the rest of the instance, survives a
// fresh pass over the same session, and keeps the message's verify stamp.
func TestCarryContentSeedsDeterministicAndStamped(t *testing.T) {
	docs := strings.Repeat("abcdefghij ", 900)
	txt := RouterReplayMessage{Role: "user", BlockTypes: []string{"text"}}
	d := ccDelta(t, txt, "")
	last := RouterReplayMessage{Role: "user", Hash: "u-cc", BlockTypes: []string{"text"}, Bytes: 1500 + d, CacheControl: "ephemeral"}
	next := RouterReplayMessage{Role: "user", Hash: "u", BlockTypes: []string{"text"}, Bytes: 1500}
	mk := func() RouterReplaySession {
		s := twoTurn(last, next)
		third := RouterReplayRequest{Messages: append(append([]RouterReplayMessage{}, s.Instances[0].Requests[1].Messages...),
			RouterReplayMessage{Role: "assistant", Hash: "a2", BlockTypes: []string{"text"}, Bytes: 400})}
		third.Messages[3].CacheControl = ""
		third.Messages[3].Hash = "tail-nocc"
		third.Messages[3].Bytes -= d
		s.Instances[0].Requests = append(s.Instances[0].Requests, third)
		return s
	}
	s1, s2 := mk(), mk()
	carryContentSeeds(&s1)
	carryContentSeeds(&s1) // idempotent
	carryContentSeeds(&s2)
	r := s1.Instances[0].Requests
	if r[2].Messages[1].contentSeed() != "u-cc" {
		t.Errorf("third request: seed %q, want u-cc", r[2].Messages[1].contentSeed())
	}
	if r[2].Messages[3].contentSeed() != "tail" {
		t.Errorf("tail losing its marker: seed %q, want tail", r[2].Messages[3].contentSeed())
	}
	su := buildSessionUUIDs(s1, "stamp")
	if _, ok := su.hashToTurn["u"]; ok {
		t.Error("carried message got a turn of its own")
	}
	p := &replayPoster{uuidEnabled: true}
	inj1, inj2 := p.buildInjection(r[0], su), p.buildInjection(r[1], su)
	if inj1 == nil || inj2 == nil {
		t.Fatal("no injection for a request carrying a qualifying turn")
	}
	c1, _ := json.Marshal(buildMessageContent(r[0].Messages[1], docs, replaySizing{}, inj1.StampByHash))
	c2, _ := json.Marshal(buildMessageContent(r[1].Messages[1], docs, replaySizing{}, inj2.StampByHash))
	if string(c1) != string(c2) {
		t.Error("stamped message differs between turns")
	}
	if !strings.Contains(string(c1), "turn-2 id: ") {
		t.Errorf("stamp missing: %s", c1)
	}
	o1 := bodies(t, s1.Instances[0].Requests[2], docs, nil)
	o2 := bodies(t, s2.Instances[0].Requests[2], docs, nil)
	for path := range o1 {
		if string(o1[path]) != string(o2[path]) {
			t.Errorf("%s: a fresh pass over the session sent different bytes", path)
		}
	}
}
