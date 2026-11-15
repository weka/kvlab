package affinity

import (
	"math"

	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/registry"
)

// causeSet records which band signals got an affinity holder skipped in one
// decision, so a split can be attributed (router_split_trigger_total) and the
// relative signals' eligibility applied to its target.
type causeSet uint8

const (
	causeConcurrency causeSet = 1 << iota
	causeSoftTTFT
	causeRelTTFT
	causeRelDecode
)

// count increments router_split_trigger_total once per cause present.
func (c causeSet) count() {
	for bit, label := range map[causeSet]string{
		causeConcurrency: "concurrency", causeSoftTTFT: "soft_ttft",
		causeRelTTFT: "relative_ttft", causeRelDecode: "relative_decode",
	} {
		if c&bit != 0 {
			metrics.SplitTriggers.WithLabelValues(label).Inc()
		}
	}
}

// lowestOther is the lowest f over usable backends other than self; ok is
// false when there is no other backend to compare against.
func lowestOther(usable []*registry.Backend, self *registry.Backend, f func(*registry.Backend) int64) (int64, bool) {
	var lo int64
	ok := false
	for _, o := range usable {
		if o == self {
			continue
		}
		if v := f(o); !ok || v < lo {
			lo, ok = v, true
		}
	}
	return lo, ok
}

// selectByCost returns the usable backend with the lowest prefill-plus-load cost
// (see Config.CostRouting). Ties go to the fewest in-flight requests. It marks
// the winner, so the tree records the copy a move creates.
//
// A prefill-only request (rr.PrefillOnly) is costed by the node's queued
// uncached prefill tokens (rr.PrefillQueue, else the node's in-flight prefill
// gauge) plus its own uncached tokens there. The node's active context is left
// out: on a node that only runs prefill legs it is the full prompt of every
// queued leg, cached part included, so it measures queue length in prompts
// rather than in work, and one extra queued leg would outweigh a whole held
// prefix.
//
// For a disaggregated leg (prefill-only, or a decode leg receiving its KV by
// transfer) the tokens a node lacks are weighted by Config.MoveWeight, so a
// holder keeps the conversation until its extra load exceeds MoveWeight times
// what moving costs: the recompute a prefill leg would put on another node's
// queue, or the KV a decode leg would pull and hold again.
func (p *Policy) selectByCost(usable []*registry.Backend, slots []int, pth path, rr *policy.RoutingRequest) *registry.Backend {
	var total int64
	for i := 0; i < pth.len(); i++ {
		total += int64(pth.tokens(i))
	}
	// local is what each node holds itself; held adds what any node can load
	// from a shared tier, which is all the cost sees.
	local := p.tree.heldTokens(pth, slots)
	held := local
	if credit := min(rr.PreloadedTokens, total); credit > 0 {
		held = make([]int64, len(local))
		for i := range held {
			held[i] = max(local[i], credit)
		}
	}
	var deepest int64
	best := 0
	var bestCost int64
	for i, c := range usable {
		deepest = max(deepest, local[i])
		var cost int64
		switch {
		case !rr.PrefillOnly:
			cost = p.moveCost(rr, total-held[i]) + c.PrefillTokens() + c.ActiveTokens()
		case rr.PrefillQueue != nil:
			cost = p.moveCost(rr, total-held[i]) + rr.PrefillQueue(c)
		default:
			cost = p.moveCost(rr, total-held[i]) + c.PrefillTokens()
		}
		if i == 0 || cost < bestCost || (cost == bestCost && p.costTieWins(rr, c, usable[best], local[i], local[best])) {
			best, bestCost = i, cost
		}
	}
	rr.PolicyState = markDecision
	rr.Prefill = &policy.PrefillEstimate{Backend: usable[best], Tokens: total - held[best]}
	if local[best] < deepest {
		p.m.Decision("cost_move")
	} else {
		p.m.Decision("cost_stay")
	}
	return usable[best]
}

// moveCost is the cost term for the tokens a node lacks: weighted by
// Config.MoveWeight for a disaggregated leg, as is for anything else.
func (p *Policy) moveCost(rr *policy.RoutingRequest, tokens int64) int64 {
	if !rr.PrefillOnly && !rr.KVTransfer {
		return tokens
	}
	return int64(math.Round(float64(tokens) * p.cfg.MoveWeight))
}

// costTieWins breaks a cost tie between candidate c and the best so far. A
// prefill-only request prefers the node holding more of its prompt itself: the
// cost counts a tier load as free, and a local hit is cheaper still, so a tie
// is no reason to leave the holder. Otherwise, and between equal holders, the
// fewest in-flight requests win.
func (p *Policy) costTieWins(rr *policy.RoutingRequest, c, best *registry.Backend, cLocal, bestLocal int64) bool {
	if rr.PrefillOnly && cLocal != bestLocal {
		return cLocal > bestLocal
	}
	return c.Inflight() < best.Inflight()
}

// estTokens is this request's estimated context size in tokens: the gateway's
// body-based estimate when present, else the sum of its routing units.
func estTokens(rr *policy.RoutingRequest) int64 {
	if rr.EstTokens > 0 {
		return rr.EstTokens
	}
	var n int64
	for _, u := range rr.Units {
		n += int64(u.Tokens)
	}
	return n
}
