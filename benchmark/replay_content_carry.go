package benchmark

import (
	"encoding/json"
	"slices"
)

// contentCarry is what a message's synthesized content is derived from: the
// seed its text is drawn by, and the captured size its blocks are split from.
// For most messages that is the message itself. A message that repeats an
// earlier one whose capture differs only in its cache_control marker carries
// the earlier one's, so both are synthesized to the same content bytes.
type contentCarry struct {
	Seed   string
	Bytes  int
	Tokens int
	// CacheControl is the marker the sized message carried: its skeleton
	// counts it, so the content budget is the same one the source had.
	CacheControl string
}

// carryContentSeeds links, within each instance of sess, every message whose
// capture differs from the message at the same position of the previous
// request only by cache_control to that earlier message's content.
//
// The capture hashes a message's canonical block JSON, cache_control
// included, so when a client moves its cache breakpoint off the previous
// turn's last message (Claude Code does on every turn), that message's hash
// changes though its content does not; the upstream cache read the whole
// previous prompt. Seeding by hash alone sends that message with new content
// on the next turn, and a server's prefix cache breaks there every turn.
//
// The carry is derived from the session's own message sequence, instance by
// instance and in file order, so every process replaying the session sends
// the same bytes, and a truncated session (a prefix of each instance) keeps
// the carries of what it kept. It is idempotent. The cache_control marker
// itself is still emitted as captured (buildMessages).
func carryContentSeeds(sess *RouterReplaySession) {
	for ii := range sess.Instances {
		reqs := sess.Instances[ii].Requests
		byHash := map[string]contentCarry{}
		var prev []RouterReplayMessage
		for ri := range reqs {
			msgs := reqs[ri].Messages
			for i := range msgs {
				m := &msgs[i]
				if m.Hash == "" {
					continue
				}
				if c, ok := byHash[m.Hash]; ok {
					m.carry = c
					continue
				}
				c := contentCarry{Seed: m.Hash, Bytes: m.Bytes, Tokens: m.Tokens, CacheControl: m.CacheControl}
				if i < len(prev) && cacheControlOnlyChange(prev[i], *m) {
					c = prev[i].contentSource()
				}
				m.carry = c
				byHash[m.Hash] = c
			}
			prev = msgs
		}
	}
}

// contentSeed is the seed m's synthesized content and its tool names are
// drawn by, and the key its verify stamp is found under.
func (m RouterReplayMessage) contentSeed() string {
	if m.carry.Seed != "" {
		return m.carry.Seed
	}
	return m.Hash
}

// contentSource is the carry m's content is synthesized from.
func (m RouterReplayMessage) contentSource() contentCarry {
	if m.carry.Seed != "" {
		return m.carry
	}
	return contentCarry{Seed: m.Hash, Bytes: m.Bytes, Tokens: m.Tokens, CacheControl: m.CacheControl}
}

// cacheControlOnlyChange reports whether b is a's captured message with only
// its cache_control marker added or removed: same role, block types and tool
// ids, a marker on exactly one of them, and captured bytes that differ by
// exactly the size of a marker the capture can have written (see
// cacheControlDeltas).
func cacheControlOnlyChange(a, b RouterReplayMessage) bool {
	if a.Hash == b.Hash || a.Role != b.Role ||
		!slices.Equal(a.BlockTypes, b.BlockTypes) ||
		!slices.Equal(a.ToolUseIDs, b.ToolUseIDs) ||
		!slices.Equal(a.ToolResultIDs, b.ToolResultIDs) {
		return false
	}
	with, without := a, b
	if with.CacheControl == "" {
		with, without = b, a
	}
	if with.CacheControl == "" || without.CacheControl != "" {
		return false
	}
	d := with.Bytes - without.Bytes
	for _, want := range cacheControlDeltas(with) {
		if d == want {
			return true
		}
	}
	return false
}

// cacheControlTTLs are the ttl values a captured cache_control can carry
// besides none. The replay file keeps only the marker's type, so each is a
// candidate for the marker's size.
var cacheControlTTLs = []string{"", "5m", "1h"}

// cacheControlDeltas lists the byte differences between m's captured
// canonical JSON and the same message without its cache_control: the
// marker's size on the last block for each possible ttl, and, for a single
// text block, that plus the block wrapper, since a client that drops the
// marker can send such a message in the string-content form. Each is
// measured on m's skeleton the way the capture measures a message.
func cacheControlDeltas(m RouterReplayMessage) []int {
	skel := messageSkeleton(m)
	if len(skel) == 0 {
		return nil
	}
	role := m.Role
	size := func(content interface{}) int {
		b, _ := json.Marshal(map[string]interface{}{"role": role, "content": content})
		return len(b)
	}
	bare := size(skel)
	var out []int
	for _, ttl := range cacheControlTTLs {
		cc := map[string]string{"type": m.CacheControl}
		if ttl != "" {
			cc["ttl"] = ttl
		}
		content := make([]map[string]interface{}, len(skel))
		copy(content, skel)
		last := make(map[string]interface{}, len(skel[len(skel)-1])+1)
		for k, v := range skel[len(skel)-1] {
			last[k] = v
		}
		last["cache_control"] = cc
		content[len(content)-1] = last
		marked := size(content)
		out = append(out, marked-bare)
		if len(m.BlockTypes) == 1 && m.BlockTypes[0] == "text" {
			out = append(out, marked-size(""))
		}
	}
	return out
}
