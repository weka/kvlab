package prefill

import (
	"slices"
	"strings"
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

// packOrderLocked sorts bs oldest prefill node first: initial nodes (never
// converted), then by when each was converted, then by URL. Under PD a job
// goes to the first with room, so the oldest nodes fill and the newest runs
// dry and is the first the release rules return; cost placement would keep
// every prefill node a little busy and none would ever go back.
func (m *Manager) packOrderLocked(bs []*registry.Backend) []*registry.Backend {
	out := slices.Clone(bs)
	slices.SortFunc(out, func(a, b *registry.Backend) int {
		na, nb := m.nodes[a.URL], m.nodes[b.URL]
		if c := na.convertedAt.Compare(nb.convertedAt); c != 0 {
			return c
		}
		return strings.Compare(a.URL, b.URL)
	})
	return out
}

// moveTargetsLocked is the nodes a waiting job may move to, in the order they
// are tried: pack order under PD, any order otherwise.
func (m *Manager) moveTargetsLocked() []*node {
	out := make([]*node, 0, len(m.nodes))
	for _, n := range m.nodes {
		out = append(out, n)
	}
	if m.cfg.PD {
		slices.SortFunc(out, func(a, b *node) int {
			if c := a.convertedAt.Compare(b.convertedAt); c != 0 {
				return c
			}
			return strings.Compare(a.b.URL, b.b.URL)
		})
	}
	return out
}

// abandonDueLocked is the delay rule's condition to abandon a drain: nothing
// waiting, or the delay signal at or below half DrainDelay. Lower than the
// drain condition, so a signal hovering at the delay does not drain and
// abandon in turn; with no hold time, so a drain no longer needed stops at
// once.
func (m *Manager) abandonDueLocked() bool {
	return m.queuedJobsLocked() == 0 || m.delayLocked() <= m.cfg.DrainDelay/2
}

// newestIdleDrainLocked is the most recently drained node with no job on it,
// or nil.
func (m *Manager) newestIdleDrainLocked() *node {
	var newest *node
	for _, n := range m.nodes {
		if n.mode == Draining && n.pending == 0 && n.running == 0 && len(n.queue) == 0 &&
			(newest == nil || n.convertedAt.After(newest.convertedAt)) {
			newest = n
		}
	}
	return newest
}

// Tick is the control loop between jobs, run every tickEvery by Run (tests
// call it directly). cands is the pool's healthy backends.
//
//   - A draining node with no job on it goes back to decode: under the budget
//     rule each such node once the overflow has cleared (overLocked); under
//     the delay rule the newest such node, one per tick, while abandonDueLocked.
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
	byDelay := m.cfg.PD && m.cfg.DrainDelay > 0
	var abandon *node
	if byDelay && nonDecode > m.cfg.MinPrefill && m.abandonDueLocked() {
		abandon = m.newestIdleDrainLocked()
	}
	for _, n := range m.nodes {
		switch {
		case n == abandon:
			m.releaseLocked(n)
			nonDecode--
		case n.mode == Prefill:
			// A node that became a prefill node with no job, or whose idle
			// start was never stamped, is idle from now.
			if n.idleSince.IsZero() && n.pending == 0 && n.running == 0 && len(n.queue) == 0 {
				n.idleSince = now
			}
		case n.mode != Draining:
		case !byDelay && !over && n.pending == 0 && len(n.queue) == 0 && n.running == 0 && nonDecode > m.cfg.MinPrefill:
			m.releaseLocked(n)
			nonDecode--
		case n.b.Inflight() == 0:
			m.setModeLocked(n, Prefill)
			m.publishLocked(n)
			m.logLocked("draining node is now a prefill node", n, "drained", "drain_took", now.Sub(n.convertedAt).Round(time.Millisecond))
		}
	}
	if m.cfg.PD && nonDecode < m.cfg.MinPrefill && len(decode) > 0 &&
		m.canConvertLocked(len(cands), nonDecode, len(decode)) {
		// Below the minimum (a prefill node released by its health, or none
		// named initially): drain toward it, one node per tick.
		m.drainLocked(decode, "min_nodes")
		nonDecode++
		decode = decode[:0]
		for _, b := range cands {
			if m.nodes[b.URL].mode == Decode {
				decode = append(decode, b)
			}
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
		if n.mode == Prefill && len(n.queue) == 0 &&
			(newest == nil || n.convertedAt.After(newest.convertedAt)) {
			newest = n
		}
	}
	if newest != nil && m.mayReleaseLocked() {
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
