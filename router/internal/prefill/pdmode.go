package prefill

import (
	"strings"

	"github.com/weka/kvlab/router/internal/registry"
)

// Dynamic vanilla P/D (Config.PD).
//
// The modes, budget, conversion, drain and release are the prefill split's,
// unchanged; what differs is which requests become jobs and what a node out of
// decode mode may serve. Every chat/completions request's prefill leg is a
// job, so the prefill load the budget measures is the whole fleet's, and a
// decode node is drained into prefill when the prefill nodes stay over budget
// for ConvertAfter. A draining node finishes its decodes before it takes a
// leg, so no node runs prefill legs alongside decodes it was given as a
// decode node. Floor nodes (--pd-prefill-backends) are prefill from the start
// and never released. A node that is draining or prefill never takes a decode
// leg, even when no decode node is left.

func normalize(u string) string { return strings.TrimRight(strings.TrimSpace(u), "/") }

// inPrefillLocked reports whether b is out of decode routing: a node in any
// mode but Decode, or a floor node the manager has not met yet.
func (m *Manager) inPrefillLocked(b *registry.Backend) bool {
	if n := m.nodes[b.URL]; n != nil {
		return n.mode != Decode
	}
	return m.floor[normalize(b.URL)]
}

// IsPrefill reports whether b is out of decode routing: draining or prefill,
// or a floor node.
func (m *Manager) IsPrefill(b *registry.Backend) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.inPrefillLocked(b)
}

// PrefillNodes keeps the members of cands that are out of decode routing.
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

// logLocked writes a conversion or release line, with the node's load and the
// prefill nodes' totals, when Config.Log is set.
func (m *Manager) logLocked(msg string, n *node, reason string, jobTokens int64) {
	if m.cfg.Log == nil {
		return
	}
	var counts [3]int
	var pending, running, queuedJobs int64
	for _, x := range m.nodes {
		counts[x.mode]++
		if x.mode != Decode {
			pending += x.pending
			running += x.runningTokens
			queuedJobs += int64(len(x.queue))
		}
	}
	attrs := []any{"backend", n.b.URL, "reason", reason, "mode", n.mode.String(),
		"node_inflight", n.b.Inflight(), "node_pending_tokens", n.pending,
		"prefill_nodes", counts[Prefill], "draining_nodes", counts[Draining], "decode_nodes", counts[Decode],
		"prefill_pending_tokens", pending, "prefill_running_tokens", running,
		"prefill_queued_tokens", pending - running, "prefill_queued_jobs", queuedJobs,
		"node_budget_tokens", m.cfg.NodeBudgetTokens}
	if jobTokens > 0 {
		attrs = append(attrs, "job_tokens", jobTokens)
	}
	m.cfg.Log.Info(msg, attrs...)
}
