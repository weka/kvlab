package prefill

import (
	"net/url"
	"sort"
	"time"

	"github.com/weka/kvlab/router/internal/registry"
)

// provisionInitialLocked makes Config.InitialNodes nodes prefill nodes in
// reject mode, from the healthy backends in cands, as they appear: with
// discovered pods the fleet grows after start, so each call takes what the
// cap and the idle nodes allow until the count is reached, then never acts
// again.
//
// A node goes straight from decode to prefill, skipping the drain, the
// ConvertAfter wait and DrainMinInterval, because it is taken only while it
// has nothing in flight: there are no decodes to finish. A busy node is left
// for a later call. Each pick goes to the host with the fewest nodes out of
// decode, ties by URL, so the set spreads across machines. A node it has
// made prefill once is never taken again, so a node released for idleness
// stays in decode unless the traffic drains it; and once any of them has been
// released the count is dropped, so a release never makes another node
// prefill in its place.
//
// The node's idle clock does not start until its first job is done: the
// first rejects can come a minute or more after the router starts, and an
// initial node released before them would be no use. From then on
// ReleaseIdle returns it to decode like any prefill node.
func (m *Manager) provisionInitialLocked(cands []*registry.Backend) {
	if m.initialLeft <= 0 || m.cfg.PD {
		return
	}
	var idle []*node
	nonDecode, decode := 0, 0
	perHost := map[string]int{}
	for _, b := range cands {
		n := m.nodeLocked(b)
		if n.mode != Decode {
			nonDecode++
			perHost[hostOf(b.URL)]++
			continue
		}
		decode++
		if !m.initialTaken[b.URL] && b.Inflight() == 0 {
			idle = append(idle, n)
		}
	}
	sort.Slice(idle, func(i, j int) bool { return idle[i].b.URL < idle[j].b.URL })
	for m.initialLeft > 0 && len(idle) > 0 && m.canConvertLocked(len(cands), nonDecode, decode) {
		best := 0
		for i, n := range idle {
			if perHost[hostOf(n.b.URL)] < perHost[hostOf(idle[best].b.URL)] {
				best = i
			}
		}
		n := idle[best]
		idle = append(idle[:best], idle[best+1:]...)
		m.initialLeft--
		m.initialTaken[n.b.URL] = true
		perHost[hostOf(n.b.URL)]++
		nonDecode++
		decode--
		m.setModeLocked(n, Prefill)
		// Zero convertedAt: an initial node counts as the oldest.
		n.convertedAt, n.idleSince, n.unproven = time.Time{}, time.Time{}, true
		m.publishLocked(n)
		// The promotion's message, so a log reader counting conversions
		// sees these too; reason=initial tells them apart.
		m.logLocked("draining node is now a prefill node", n, "initial", "initial_left", m.initialLeft)
	}
}

// hostOf is the URL's host without its port: the machine a backend runs on.
func hostOf(u string) string {
	if p, err := url.Parse(u); err == nil && p.Hostname() != "" {
		return p.Hostname()
	}
	return u
}
