package gateway_test

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"testing"

	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/registry"
)

// These replay multi-turn conversations through the full P/D gateway under
// cost routing, against a fleet with no shared KV tier: a P node that does not
// hold a conversation's prefix recomputes all of it.

const (
	convSysChars  = 32_000 // ~8k-token conversation prefix
	convTurnChars = 2_000  // ~500 tokens per message
	convTokens    = convSysChars / 4
	turnTokens    = convTurnChars / 4
)

// conversationTurn is turn k of conversation s: a unique system prompt, then k
// earlier exchanges, then the new question. Each turn extends the previous
// turn's prompt.
func conversationTurn(s, k int) []byte {
	msgs := []map[string]any{{"role": "system", "content": fmt.Sprintf("conversation %04d ", s) + strings.Repeat(fmt.Sprintf("s%d ", s), convSysChars/4)}}
	for j := 0; j <= k; j++ {
		msgs = append(msgs, map[string]any{"role": "user", "content": fmt.Sprintf("question %d ", j) + strings.Repeat("q ", convTurnChars/2)})
		if j < k {
			msgs = append(msgs, map[string]any{"role": "assistant", "content": fmt.Sprintf("answer %d ", j) + strings.Repeat("a ", convTurnChars/2)})
		}
	}
	b, _ := json.Marshal(map[string]any{"model": "m", "max_tokens": 4, "messages": msgs})
	return b
}

// pdSaturation keeps every node of a P/D fleet busy, as a saturated fleet is.
// Before each request it redraws, per node, the other conversations' legs in
// flight there: each conversation the node served last has a leg in flight
// with probability busy, and every node also runs base±jitter legs of traffic
// outside the replay. A P leg in flight is a prefill of ctxTokens of which only
// turnTokens are uncached — a holder serving a next turn — and a D leg is a
// decode of ctxTokens.
//
// Because a node's queue grows with the conversations placed on it, piling
// conversations onto one node makes it slower, as it does in a real fleet.
type pdSaturation struct {
	f            *pdFleet
	rnd          *rand.Rand
	busy         float64
	base, jitter int
	ctxTokens    int64
	held         []*lease.Lease
	// extraTokens, when set, adds a prefill of this many uncached tokens to P
	// node extraNode's queue.
	extraNode   int
	extraTokens int64
}

func (s *pdSaturation) release() {
	for _, l := range s.held {
		l.Release()
	}
	s.held = s.held[:0]
}

// redraw sets every node's in-flight legs. owner[i] is the node that served
// conversation i's last turn on the leg set nodeOf picks, or -1; skip is the
// conversation about to be sent, which has nothing in flight.
func (s *pdSaturation) redraw(lastP, lastD []int, skip int) {
	s.release()
	legs := make([]int, len(s.f.backend))
	for i, b := range s.f.backend {
		legs[i] = s.base
		if s.jitter > 0 {
			legs[i] += s.rnd.Intn(2*s.jitter+1) - s.jitter
		}
		if s.extraTokens > 0 && i == s.extraNode {
			s.held = append(s.held, s.load(b, true, s.extraTokens))
		}
	}
	for c := range lastP {
		if c == skip {
			continue
		}
		for _, n := range []int{lastP[c], lastD[c]} {
			if n >= 0 && s.rnd.Float64() < s.busy {
				legs[n]++
			}
		}
	}
	for i, b := range s.f.backend {
		for range legs[i] {
			s.held = append(s.held, s.load(b, s.f.isP(i), turnTokens))
		}
	}
}

func (s *pdSaturation) load(b *registry.Backend, prefill bool, uncached int64) *lease.Lease {
	l := lease.Acquire(b)
	l.StartActive(s.ctxTokens)
	if prefill {
		l.StartPrefill(uncached)
	} else {
		l.StartDecode(s.ctxTokens)
	}
	return l
}

type stickiness struct {
	pStay, dStay, moves int
	pPrompt, pCached    int64
	pLegs, dLegs        []int
}

func (s stickiness) pRate() float64 { return float64(s.pStay) / float64(s.moves) }
func (s stickiness) dRate() float64 { return float64(s.dStay) / float64(s.moves) }
func (s stickiness) pHit() float64  { return float64(s.pCached) / float64(s.pPrompt) }

// replayConversations runs nConv conversations of nTurns turns through the
// gateway, interleaved turn by turn, and reports how often a turn's legs went
// to the node that served the conversation's previous turn.
func replayConversations(t *testing.T, f *pdFleet, sat *pdSaturation, nConv, nTurns int) stickiness {
	t.Helper()
	lastP := make([]int, nConv)
	lastD := make([]int, nConv)
	for i := range nConv {
		lastP[i], lastD[i] = -1, -1
	}
	st := stickiness{pLegs: make([]int, f.nP), dLegs: make([]int, len(f.backend)-f.nP)}
	for k := range nTurns {
		for s := range nConv {
			if sat != nil {
				sat.redraw(lastP, lastD, s)
			}
			before := len(f.calls)
			status, body := f.post(t, "/v1/chat/completions", string(conversationTurn(s, k)))
			if status != http.StatusOK {
				t.Fatalf("conversation %d turn %d: %d %s", s, k, status, body)
			}
			p, d := -1, -1
			for _, c := range f.calls[before:] {
				if c.isPrefillLeg() {
					p = c.node
				} else {
					d = c.node
				}
			}
			if p < 0 || d < 0 {
				t.Fatalf("conversation %d turn %d: legs P=%d D=%d", s, k, p, d)
			}
			st.pLegs[p]++
			st.dLegs[d-f.nP]++
			if k > 0 {
				st.moves++
				if p == lastP[s] {
					st.pStay++
				}
				if d == lastD[s] {
					st.dStay++
				}
			}
			lastP[s], lastD[s] = p, d
		}
	}
	if sat != nil {
		sat.release()
	}
	for i := range f.nP {
		es := f.servers[i].Engine().Stats()
		st.pPrompt += es.PromptTokens
		st.pCached += es.CachedTokens
	}
	return st
}

// Under cost routing with every P node saturated, a conversation's next turn
// must go to the P node holding its prefix: any other P node recomputes the
// whole prompt — ~16x the turn — while the holder's queue is no longer than
// anyone else's, give or take a few turns. Conversations must still spread
// evenly over the P nodes rather than pile onto a few.
func TestPDCostRoutingKeepsConversationsOnTheirPNodeUnderSaturation(t *testing.T) {
	usePDCostRouting(t, 0)
	const nP, nD, nConv, nTurns = 8, 8, 32, 6
	f := newPDFleet(t, nP, nD, "")
	sat := &pdSaturation{
		f: f, rnd: rand.New(rand.NewSource(1)),
		busy: 0.5, base: 4, jitter: 2, ctxTokens: 10_000,
	}
	st := replayConversations(t, f, sat, nConv, nTurns)
	t.Logf("P stays %d/%d (%.2f), D stays %d/%d (%.2f), P cache hit %.3f, legs per P node %v, per D node %v",
		st.pStay, st.moves, st.pRate(), st.dStay, st.moves, st.dRate(), st.pHit(), st.pLegs, st.dLegs)
	if st.pRate() < 0.95 {
		t.Errorf("P-leg stickiness %.2f: turns leave the P node holding their prefix for a node that must recompute it", st.pRate())
	}
	for i, n := range st.pLegs {
		if share := nConv * nTurns / nP; n > 2*share {
			t.Errorf("P node %d served %d legs, more than twice its share %d: conversations herded onto it (%v)", i, n, share, st.pLegs)
		}
	}
}

// The holder is abandoned only when its queue would delay the turn by more
// than a non-holder takes to recompute the whole prompt.
func TestPDCostRoutingMovesAConversationOffAFarMoreLoadedPNode(t *testing.T) {
	usePDCostRouting(t, 0)
	f := newPDFleet(t, 2, 1, "")
	sat := &pdSaturation{f: f, rnd: rand.New(rand.NewSource(1)), ctxTokens: 10_000}
	none := []int{}

	post := func(k int) int {
		before := len(f.calls)
		if status, body := f.post(t, "/v1/chat/completions", string(conversationTurn(0, k))); status != http.StatusOK {
			t.Fatalf("turn %d: %d %s", k, status, body)
		}
		for _, c := range f.calls[before:] {
			if c.isPrefillLeg() {
				return c.node
			}
		}
		t.Fatalf("turn %d ran no prefill leg", k)
		return -1
	}
	holder := post(0)

	// A queue somewhat shorter than the prefix: staying is still faster.
	sat.extraNode, sat.extraTokens = holder, convTokens/2
	sat.redraw(none, none, -1)
	if got := post(1); got != holder {
		t.Errorf("moved off the holder with %d queued tokens, less than the ~%d-token prefix", sat.extraTokens, convTokens)
	}
	// A queue several prefixes long: recomputing elsewhere is faster.
	sat.extraTokens = 5 * convTokens
	sat.redraw(none, none, -1)
	if got := post(2); got == holder {
		t.Errorf("stayed on the holder behind %d queued tokens, more than recomputing the prefix", sat.extraTokens)
	}
	sat.release()
}

// With the fleet idle only the prefix decides, so every turn of a conversation
// stays on the P node and the D node that served its first turn: the P leg's
// prefix is recorded against the P node that served it, and the next turn,
// sent with different max_tokens and without kv_transfer_params, finds it.
func TestPDCostRoutingKeepsConversationsOnTheirNodesWhenIdle(t *testing.T) {
	usePDCostRouting(t, 0)
	f := newPDFleet(t, 4, 4, "")
	st := replayConversations(t, f, nil, 8, 4)
	t.Logf("P stays %d/%d, D stays %d/%d, P cache hit %.3f", st.pStay, st.moves, st.dStay, st.moves, st.pHit())
	if st.pStay != st.moves || st.dStay != st.moves {
		t.Errorf("idle fleet moved conversations: P stays %d/%d, D stays %d/%d", st.pStay, st.moves, st.dStay, st.moves)
	}
}
