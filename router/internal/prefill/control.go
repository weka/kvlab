package prefill

import (
	"time"

	"github.com/weka/kvlab/router/internal/registry"
)

// tickEvery paces Tick under Run.
const tickEvery = 100 * time.Millisecond

// noteWaitLocked records a job's router wait when it is granted its slot.
func (m *Manager) noteWaitLocked(j *Job) {
	m.m.JobWait.Observe(m.cfg.Clock.Now().Sub(j.reservedAt).Seconds())
}

// overLocked reports whether the prefill nodes are still overflowing: what a
// drain is for, and what keeps a drain going with no job on it. A job waiting
// anywhere, or a prefill node over its budget.
func (m *Manager) overLocked() bool {
	for _, n := range m.nodes {
		if len(n.queue) > 0 || n.mode == Prefill && n.pending > m.cfg.NodeBudgetTokens {
			return true
		}
	}
	return false
}

// Tick is the control loop between jobs, run every tickEvery by Run (tests
// call it directly). cands is the pool's healthy backends. Under PD it is
// tickPD; otherwise:
//
//   - InitialNodes not yet provisioned are taken from the idle decode nodes
//     (initial.go).
//   - A draining node with no job on it goes back to decode once the overflow
//     has cleared (overLocked).
//   - Otherwise a draining node whose decodes have finished becomes a prefill
//     node, and jobs waiting elsewhere move to it.
//   - Prefill nodes idle for ReleaseIdle go back to decode, a node that has
//     never had a job counting as idle from the first Tick that sees it so;
//     an InitialNodes node only from the end of its first job.
func (m *Manager) Tick(cands []*registry.Backend) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cfg.PD {
		m.tickPDLocked(cands)
		return
	}
	m.provisionInitialLocked(cands)
	now := m.cfg.Clock.Now()
	nonDecode := 0
	for _, b := range cands {
		if m.nodeLocked(b).mode != Decode {
			nonDecode++
		}
	}
	over := m.overLocked()
	for _, n := range m.nodes {
		switch {
		case n.mode == Prefill:
			// A node that became a prefill node with no job, or whose idle
			// start was never stamped, is idle from now.
			if n.idleSince.IsZero() && !n.unproven && n.pending == 0 && n.running == 0 && len(n.queue) == 0 {
				n.idleSince = now
			}
		case n.mode != Draining:
		case !over && n.pending == 0 && len(n.queue) == 0 && n.running == 0 && nonDecode > m.cfg.MinPrefill:
			m.releaseLocked(n)
			nonDecode--
		case n.b.Inflight() == 0:
			m.promoteLocked(n, now)
		}
	}
	if m.cfg.ReleaseIdle > 0 {
		m.releaseExpiredLocked()
	}
	m.spreadLocked()
	m.publishCountsLocked()
}

// promoteLocked makes a drained node a prefill node.
func (m *Manager) promoteLocked(n *node, now time.Time) {
	m.setModeLocked(n, Prefill)
	m.publishLocked(n)
	m.logLocked("draining node is now a prefill node", n, "drained", "drain_took", now.Sub(n.convertedAt).Round(time.Millisecond))
}

func (m *Manager) queuedJobsLocked() (q int) {
	for _, n := range m.nodes {
		q += len(n.queue)
	}
	return q
}
