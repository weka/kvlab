package prefill

import (
	"strings"
	"time"

	"github.com/weka/kvlab/router/internal/registry"
)

// The elastic P pool (Config.PD).
//
// Every request's prefill leg is a job, so the P set's load is the manager's
// pending tokens: legs queued in the router waiting for budget room, plus legs
// running on a P node. The set is
//
//   - the floor: Config.Floor, always prefill, never released;
//   - converted nodes: decode nodes moved in, to reach Config.MinPrefill or
//     because the whole set has been over its budget for ConvertAfter.
//
// A converted node is Converting at first. Workers are symmetric (every
// engine is kv_both), so it takes prefill legs at once; it only stops getting
// decode legs, and becomes Prefill when the decodes it was already running
// have finished. Nothing waits for that.
//
// A converted node goes back to decode when it has had no leg for ReleaseIdle,
// or when the set's pending tokens have stayed at or below ReleaseBelow of the
// budget the set would have without it, for ReleaseIdle. Conversion needs the
// set over 100% of its budget, release under ReleaseBelow of a smaller one,
// each held for its own duration, and the low-water clock restarts at every
// role change: that gap is the hysteresis that keeps a node from flapping.
// Its legs still running finish where they are; a node with legs queued on it
// is not released.
//
// The cap is MaxFleetFraction of the healthy pool, floor included, and the
// last decode node is never converted.

func normalize(u string) string { return strings.TrimRight(strings.TrimSpace(u), "/") }

func available(bs []*registry.Backend) []*registry.Backend {
	out := make([]*registry.Backend, 0, len(bs))
	for _, b := range bs {
		if b.Available() {
			out = append(out, b)
		}
	}
	return out
}

// IsPrefill reports whether b is in the P set: floor, converting or prefill.
func (m *Manager) IsPrefill(b *registry.Backend) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.inPrefillLocked(b)
}

// PrefillNodes keeps the members of cands that are in the P set.
func (m *Manager) PrefillNodes(cands []*registry.Backend) []*registry.Backend {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*registry.Backend, 0, len(cands))
	for _, b := range cands {
		if m.inPrefillLocked(b) {
			out = append(out, b)
		}
	}
	return out
}

// Step is the PD pool's control loop, run between jobs (Run calls it on a
// ticker, tests call it directly): reach the minimum, promote converting nodes
// whose decodes are done, convert for an overflow that has lasted
// ConvertAfter with legs waiting, and release idle or surplus converted
// nodes. cands is the pool's healthy backends.
func (m *Manager) Step(cands []*registry.Backend) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, b := range cands {
		m.nodeLocked(b)
	}
	m.ensureMinLocked(cands)
	for _, n := range m.nodes {
		m.promoteLocked(n)
	}
	m.convertIfOverLocked(cands)
	if m.cfg.ReleaseIdle > 0 {
		m.releaseExpiredLocked()
	}
	m.releaseLowLocked()
	m.spreadLocked()
	m.publishCountsLocked()
}

// stepEvery paces Step: a quarter of the shorter of the two hold times,
// between 50ms and 1s.
func (m *Manager) stepEvery() time.Duration {
	d := time.Second
	for _, h := range []time.Duration{m.cfg.ConvertAfter, m.cfg.ReleaseIdle} {
		if h > 0 {
			d = min(d, h/4)
		}
	}
	return max(d, 50*time.Millisecond)
}

// countsLocked splits cands into the P set and decode nodes.
func (m *Manager) countsLocked(cands []*registry.Backend) (nonDecode, decode []*registry.Backend) {
	for _, b := range cands {
		if m.nodeLocked(b).mode == Decode {
			decode = append(decode, b)
		} else {
			nonDecode = append(nonDecode, b)
		}
	}
	return nonDecode, decode
}

// ensureMinLocked converts decode nodes until the P set holds MinPrefill of
// cands, within the cap.
func (m *Manager) ensureMinLocked(cands []*registry.Backend) {
	if !m.cfg.PD || m.cfg.MinPrefill <= 0 {
		return
	}
	nonDecode, decode := m.countsLocked(cands)
	for len(nonDecode) < m.cfg.MinPrefill && m.canConvertLocked(len(cands), len(nonDecode), len(decode)) {
		b := m.leastLoadedLocked(decode)
		m.convertLocked(m.nodes[b.URL], "min", len(cands))
		nonDecode = append(nonDecode, b)
		decode = without(decode, b)
	}
}

func without(bs []*registry.Backend, x *registry.Backend) []*registry.Backend {
	out := make([]*registry.Backend, 0, len(bs))
	for _, b := range bs {
		if b != x {
			out = append(out, b)
		}
	}
	return out
}

// convertIfOverLocked converts a decode node when legs are waiting in the
// router and the P set has been over budget for ConvertAfter: the same rule
// Reserve applies to a new leg, applied to legs already queued, which then
// spread onto the new node.
func (m *Manager) convertIfOverLocked(cands []*registry.Backend) {
	if !m.cfg.PD || m.queuedJobsLocked() == 0 {
		return
	}
	nonDecode, decode := m.countsLocked(cands)
	if len(nonDecode) == 0 || !m.canConvertLocked(len(cands), len(nonDecode), len(decode)) || !m.overflowedLongEnoughLocked(nonDecode) {
		return
	}
	for _, b := range nonDecode {
		if n := m.nodes[b.URL]; n.pending == 0 || n.pending < m.cfg.NodeBudgetTokens {
			return // a node with room: the queue will drain onto it
		}
	}
	m.convertLocked(m.nodes[m.leastLoadedLocked(decode).URL], "over_budget", len(cands))
}

func (m *Manager) queuedJobsLocked() (q int) {
	for _, n := range m.nodes {
		q += len(n.queue)
	}
	return q
}

// promoteLocked makes a converting node Prefill once it runs nothing but
// prefill legs; one converted with no decode running is Prefill at the next
// Step.
func (m *Manager) promoteLocked(n *node) {
	if n.mode == Converting && n.b.Inflight() <= int64(n.running) {
		m.setModeLocked(n, Prefill)
		m.publishLocked(n)
	}
}

// mayReleaseLocked reports whether the P set may lose a node and keep
// MinPrefill.
func (m *Manager) mayReleaseLocked() bool {
	k := 0
	for _, n := range m.nodes {
		if n.mode != Decode && n.b.Available() {
			k++
		}
	}
	return k-1 >= m.cfg.MinPrefill
}

// releaseLowLocked releases one converted node once the P set's pending tokens
// have stayed at or below ReleaseBelow of the budget of the set without one
// node, for ReleaseIdle: the node with the fewest pending tokens and nothing
// queued, the latest converted on a tie. The clock restarts at it, so the
// next release needs a full window of its own.
func (m *Manager) releaseLowLocked() {
	if !m.cfg.PD || m.cfg.ReleaseBelow <= 0 {
		return
	}
	var set int
	var pending int64
	var victim *node
	for _, n := range m.nodes {
		if n.mode == Decode || !n.b.Available() {
			continue
		}
		set++
		pending += n.pending
		if n.floor || len(n.queue) > 0 {
			continue
		}
		if victim == nil || n.pending < victim.pending ||
			n.pending == victim.pending && n.convertedAt.After(victim.convertedAt) {
			victim = n
		}
	}
	low := float64(set-1) * float64(m.cfg.NodeBudgetTokens) * m.cfg.ReleaseBelow
	if victim == nil || set-1 < max(m.cfg.MinPrefill, 1) || float64(pending) > low {
		m.lowSince = time.Time{}
		return
	}
	now := m.cfg.Clock.Now()
	if m.lowSince.IsZero() {
		m.lowSince = now
	}
	if now.Sub(m.lowSince) < m.cfg.ReleaseIdle {
		return
	}
	m.releaseLocked(victim, "low_water")
	m.lowSince = now
}

// loadAttrsLocked is the P set's load, for a conversion or release log line.
func (m *Manager) loadAttrsLocked(fleet int) []any {
	var counts [4]int
	var pending, running, queuedJobs int64
	for _, n := range m.nodes {
		counts[n.mode]++
		pending += n.pending
		running += n.runningTokens
		queuedJobs += int64(len(n.queue))
	}
	attrs := []any{
		"prefill_nodes", counts[Prefill], "converting_nodes", counts[Converting],
		"draining_nodes", counts[Draining], "decode_nodes", counts[Decode],
		"pending_tokens", pending, "running_tokens", running, "queued_tokens", pending - running,
		"queued_jobs", queuedJobs, "node_budget_tokens", m.cfg.NodeBudgetTokens,
	}
	if fleet > 0 {
		attrs = append(attrs, "fleet", fleet, "max_prefill_nodes", min(int(float64(fleet)*m.cfg.MaxFleetFraction), fleet-1))
	}
	return attrs
}
