package prefill

import (
	"strings"

	"github.com/weka/kvlab/router/internal/registry"
)

// Dynamic vanilla P/D (Config.PD).
//
// The modes, drain and release are the prefill split's; what differs is which
// requests become jobs, what drives a drain, and what a node out of decode
// mode may serve. Every chat/completions request's prefill leg is a job. With
// DrainDelay set a decode node is drained when the legs' router wait stays
// high (see control.go): a P node's work is mostly loading KV from the shared
// tier, which a compute-token budget does not see, while the time a leg waits
// for a P node does. A draining node finishes its decodes before it takes a
// leg, so no node runs prefill legs alongside decodes it was given as a
// decode node. Floor nodes (--pd-prefill-backends) are prefill from the start
// and never released. A node that is draining or prefill never takes a decode
// leg, even when no decode node is left.

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
func (m *Manager) logLocked(msg string, n *node, reason string, extra ...any) {
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
	m.cfg.Log.Info(msg, append(attrs, extra...)...)
}
