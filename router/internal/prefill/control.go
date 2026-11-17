package prefill

import (
	"slices"
	"time"

	"github.com/weka/kvlab/router/internal/registry"
)

// tickEvery paces Tick under Run.
const tickEvery = 100 * time.Millisecond

// waitWindow is how far back the delay signal looks at finished waits.
const waitWindow = 10 * time.Second

type waitSample struct {
	at time.Time
	d  time.Duration
}

// noteWaitLocked records a job's router wait when it is granted its slot.
func (m *Manager) noteWaitLocked(j *Job) {
	now := m.cfg.Clock.Now()
	d := now.Sub(j.reservedAt)
	m.m.JobWait.Observe(d.Seconds())
	m.waits = append(m.waits, waitSample{at: now, d: d})
	m.trimWaitsLocked(now)
}

func (m *Manager) trimWaitsLocked(now time.Time) {
	i := 0
	for i < len(m.waits) && now.Sub(m.waits[i].at) > waitWindow {
		i++
	}
	m.waits = m.waits[i:]
}

// delayLocked is the delay signal: the 90th percentile of the router waits of
// jobs granted in the last waitWindow and of the jobs still waiting, each
// counted at its age so far. A backlog shows at once, before any of it is
// granted.
func (m *Manager) delayLocked() time.Duration {
	now := m.cfg.Clock.Now()
	m.trimWaitsLocked(now)
	ds := make([]time.Duration, 0, len(m.waits))
	for _, w := range m.waits {
		ds = append(ds, w.d)
	}
	for _, n := range m.nodes {
		for _, j := range n.queue {
			ds = append(ds, now.Sub(j.reservedAt))
		}
	}
	if len(ds) == 0 {
		return 0
	}
	slices.Sort(ds)
	return ds[(len(ds)*9+9)/10-1]
}

// overLocked reports whether the prefill nodes are still overflowing: what a
// drain is for, and what keeps a drain going with no job on it. Under the
// delay rule, jobs waiting now and the delay signal above DrainDelay: the
// signal's window still holds a past backlog's waits after it has cleared,
// and a fleet with nothing waiting needs no more prefill nodes. Under the
// budget rule, a job waiting anywhere or a prefill node over its budget.
func (m *Manager) overLocked() bool {
	if m.cfg.PD && m.cfg.DrainDelay > 0 {
		return m.queuedJobsLocked() > 0 && m.delayLocked() > m.cfg.DrainDelay
	}
	for _, n := range m.nodes {
		if len(n.queue) > 0 || n.mode == Prefill && n.pending > m.cfg.NodeBudgetTokens {
			return true
		}
	}
	return false
}

// lowLocked reports whether the prefill load is low enough to spare the newest
// converted node: under the delay rule, the signal at or below its low
// watermark. The budget rule has no such signal and spares nothing.
func (m *Manager) lowLocked() bool {
	return m.cfg.PD && m.cfg.DrainDelay > 0 && !m.delayLow.IsZero()
}

// spareNewestLocked leaves the most recently converted node out of eligible
// under cost placement and the delay rule, while another eligible node exists
// and the load is low. Work then goes to the floor and the oldest converted nodes first, and
// the newest runs dry, so the release rules can return it to decode; cost
// placement would otherwise keep every prefill node a little busy and none
// would ever go back. The ladder needs no help: it packs onto the fullest.
func (m *Manager) spareNewestLocked(eligible []*registry.Backend) []*registry.Backend {
	if !m.cfg.CostPick || len(eligible) < 2 || !m.lowLocked() {
		return eligible
	}
	var newest *node
	for _, b := range eligible {
		n := m.nodes[b.URL]
		if !n.floor && (newest == nil || n.convertedAt.After(newest.convertedAt)) {
			newest = n
		}
	}
	if newest == nil {
		return eligible
	}
	out := make([]*registry.Backend, 0, len(eligible)-1)
	for _, b := range eligible {
		if b != newest.b {
			out = append(out, b)
		}
	}
	return out
}

// Tick is the control loop between jobs, run every tickEvery by Run (tests
// call it directly). cands is the pool's healthy backends.
//
//   - A draining node with no job on it goes back to decode once the overflow
//     has cleared (overLocked).
//   - Otherwise a draining node whose decodes have finished becomes a prefill
//     node, and jobs waiting elsewhere move to it.
//   - Under the delay rule: jobs waiting and the delay signal above DrainDelay
//     for ConvertAfter drains a decode node (within the cap and
//     DrainMinInterval), and the clock restarts. Nothing waiting, or the
//     signal at or below DrainDelay/4, for ReleaseIdle, returns the newest
//     converted prefill node with nothing queued to decode, and that clock
//     restarts too.
//   - Prefill nodes idle for ReleaseIdle go back to decode, a node that has
//     never had a job counting as idle from the first Tick that sees it so.
func (m *Manager) Tick(cands []*registry.Backend) {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.cfg.Clock.Now()
	var decode []*registry.Backend
	nonDecode := 0
	for _, b := range cands {
		if m.nodeLocked(b).mode == Decode {
			decode = append(decode, b)
		} else {
			nonDecode++
		}
	}
	over := m.overLocked()
	for _, n := range m.nodes {
		switch {
		case n.mode == Prefill:
			// A node that became a prefill node with no job, or whose idle
			// start was never stamped, is idle from now.
			if !n.floor && n.idleSince.IsZero() && n.pending == 0 && n.running == 0 && len(n.queue) == 0 {
				n.idleSince = now
			}
		case n.mode != Draining:
		case !over && n.pending == 0 && len(n.queue) == 0 && n.running == 0:
			m.releaseLocked(n)
			nonDecode--
		case n.b.Inflight() == 0:
			m.setModeLocked(n, Prefill)
			m.publishLocked(n)
			m.logLocked("draining node is now a prefill node", n, "drained", "drain_took", now.Sub(n.convertedAt).Round(time.Millisecond))
		}
	}
	if m.cfg.PD && m.cfg.DrainDelay > 0 {
		m.delayRuleLocked(now, cands, decode, nonDecode)
	}
	if m.cfg.ReleaseIdle > 0 {
		m.releaseExpiredLocked()
	}
	m.spreadLocked()
	m.publishCountsLocked()
}

func (m *Manager) delayRuleLocked(now time.Time, cands, decode []*registry.Backend, nonDecode int) {
	d := m.delayLocked()
	if d > m.cfg.DrainDelay && m.queuedJobsLocked() > 0 {
		m.delayLow = time.Time{}
		if m.delayOver.IsZero() {
			m.delayOver = now
		}
		if now.Sub(m.delayOver) >= m.cfg.ConvertAfter && len(decode) > 0 &&
			m.canConvertLocked(len(cands), nonDecode, len(decode)) && m.drainAllowedLocked(nonDecode) {
			m.drainLocked(decode, "delay_over", "delay_p90", d.Round(time.Millisecond),
				"over_for", now.Sub(m.delayOver).Round(time.Millisecond))
			m.delayOver = now
		}
		return
	}
	m.delayOver = time.Time{}
	if d > m.cfg.DrainDelay/4 && m.queuedJobsLocked() > 0 {
		m.delayLow = time.Time{}
		return
	}
	if m.delayLow.IsZero() {
		m.delayLow = now
	}
	if now.Sub(m.delayLow) < m.cfg.ReleaseIdle {
		return
	}
	var newest *node
	for _, n := range m.nodes {
		if n.mode == Prefill && !n.floor && len(n.queue) == 0 &&
			(newest == nil || n.convertedAt.After(newest.convertedAt)) {
			newest = n
		}
	}
	if newest != nil {
		m.releaseForLocked(newest, "low_delay")
		m.delayLow = now
	}
}

func (m *Manager) queuedJobsLocked() (q int) {
	for _, n := range m.nodes {
		q += len(n.queue)
	}
	return q
}
