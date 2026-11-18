package prefill

import (
	"strings"
	"time"

	"github.com/weka/kvlab/router/internal/registry"
)

// TTFT-driven provisioning for dynamic P/D (Config.PD).
//
// Two mechanisms, kept apart:
//
//   - Routing (PickPrefill): every prefill leg goes at once to an active prefill
//     node, with no queue and no budget, by cost in seconds: the node's
//     estimated time for a leg (est) plus UnknownCostPer1K for each 1,000
//     prompt tokens the node is not known to hold. Known prefix is free.
//     Active is a prefill node that is not returning to decode; a node still
//     draining into prefill takes no leg until its decodes have finished.
//   - Provisioning (tickPDLocked): the mean time to the prefill's finish
//     (arrival to the P leg's kv_transfer_params) of the last TTFTWindow P/D
//     requests (or of the last TTFTWindowTime): the part of TTFT the prefill
//     set controls. Above TTFTHigh one decode node is drained into
//     prefill; below TTFTLow the newest drain is abandoned, or else one
//     prefill node is selected to return to decode: it takes no new leg,
//     finishes its in-flight ones, then becomes a decode node. A window with
//     no recent request takes no action: no traffic says nothing about the
//     prefill set, and a router up before its clients must not release the
//     initial nodes. After an action the next waits for
//     DrainMinInterval and for TTFTWindow fresh requests, so one window does
//     not drive several actions. MinPrefill and MaxFleetFraction bound it.

const (
	// legParallelism is how many legs a P node runs before each slows the
	// others: est grows by 1/legParallelism of a leg's time per leg in flight.
	legParallelism = 4
	// svcStale is how long a node's own service EWMA counts without a fresh
	// sample before est falls back to the fleet's.
	svcStale = 10 * time.Second
	// ttftStale is how long the TTFT window counts without a new request;
	// after it the window reads as no data, which is low.
	ttftStale = 10 * time.Second
	// svcAlpha and fleetAlpha are the EWMA weights of a node's and the
	// fleet's newest service sample.
	svcAlpha, fleetAlpha = 0.2, 0.05
	// minWindowSamples is the fewest samples a time window needs before its
	// mean is acted on: a window with a few requests in it is too thin to
	// drain or release on, and holds, as an empty one does.
	minWindowSamples = 16
)

// ttftWindow is the provisioning samples: the last N in a ring (count mode),
// or those of the last Config.TTFTWindowTime, since the last action (time
// mode).
type ttftWindow struct {
	buf   []time.Duration
	next  int
	full  bool
	since int       // samples since the last provisioning action
	last  time.Time // when the newest sample arrived
	timed []timedSample
}

type timedSample struct {
	at time.Time
	d  time.Duration
}

func (w *ttftWindow) addTimed(d, span time.Duration, now time.Time) {
	w.timed = append(w.timed, timedSample{at: now, d: d})
	w.pruneTimed(span, now)
	w.since++
	w.last = now
}

func (w *ttftWindow) pruneTimed(span time.Duration, now time.Time) {
	i := 0
	for i < len(w.timed) && now.Sub(w.timed[i].at) > span {
		i++
	}
	if i > 0 {
		w.timed = append(w.timed[:0], w.timed[i:]...)
	}
}

// meanTimed is the mean and size of the samples in the last span. It keeps
// older samples: a shorter span is read while a longer one is stored.
func (w *ttftWindow) meanTimed(span time.Duration, now time.Time) (time.Duration, int) {
	var sum time.Duration
	k := 0
	for _, s := range w.timed {
		if now.Sub(s.at) <= span {
			sum += s.d
			k++
		}
	}
	if k == 0 {
		return 0, 0
	}
	return sum / time.Duration(k), k
}

func (w *ttftWindow) add(d time.Duration, n int, now time.Time) {
	if len(w.buf) != n {
		w.buf, w.next, w.full = make([]time.Duration, n), 0, false
	}
	w.buf[w.next] = d
	w.next = (w.next + 1) % n
	w.full = w.full || w.next == 0
	w.since++
	w.last = now
}

// mean is the window's mean and size; 0, 0 when it holds nothing recent.
func (w *ttftWindow) mean(now time.Time) (time.Duration, int) {
	k := w.next
	if w.full {
		k = len(w.buf)
	}
	if k == 0 || now.Sub(w.last) > ttftStale {
		return 0, 0
	}
	var sum time.Duration
	for _, d := range w.buf[:k] {
		sum += d
	}
	return sum / time.Duration(k), k
}

// ObserveTTFT records a P/D request's time to the prefill's finish: arrival to
// its P leg's kv_transfer_params. Only legs that handed off their KV count.
func (m *Manager) ObserveTTFT(d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cfg.TTFTWindowTime > 0 {
		m.ttft.addTimed(d, max(m.cfg.TTFTWindowTime, m.cfg.UnderutilWindowTime), m.cfg.Clock.Now())
		return
	}
	m.ttft.add(d, m.cfg.TTFTWindow, m.cfg.Clock.Now())
}

// windowMeanLocked is the prefill-finish signal over span (the count window
// when span is 0) and its sample count.
func (m *Manager) windowMeanLocked(now time.Time, span time.Duration) (time.Duration, int) {
	if span > 0 {
		return m.ttft.meanTimed(span, now)
	}
	return m.ttft.mean(now)
}

// snapshot is everything one provisioning decision reads, taken once per
// tick: the decision and the published gauges see the same values.
type snapshot struct {
	// span is the window in force: --pd-underutil-window-time while the
	// decode side is underutilized, else --pd-ttft-window-time; 0 is the
	// count window.
	span    time.Duration
	mean    time.Duration
	samples int
	decode  decodeSignals
	under   decodeSignals // over the underutilization window
}

// snapshotLocked takes the provisioning snapshot. Underutilization is judged
// on the short window, and while it holds that window is the one in force.
func (m *Manager) snapshotLocked(now time.Time) snapshot {
	s := snapshot{span: m.cfg.TTFTWindowTime}
	s.under = m.decodeSignalsLocked(now, m.underSpan())
	if s.under.under && s.span > 0 && m.cfg.UnderutilWindowTime > 0 {
		s.span = m.cfg.UnderutilWindowTime
	}
	ds := s.span
	if ds == 0 {
		ds = defaultStarveSpan
	}
	s.decode = m.decodeSignalsLocked(now, ds)
	s.mean, s.samples = m.windowMeanLocked(now, s.span)
	return s
}

func (m *Manager) publishSnapshotLocked(s snapshot) {
	pm := m.cfg.PDMetrics
	if pm == nil {
		return
	}
	pm.TTFTMean.Set(s.mean.Seconds())
	pm.TTFTSamples.Set(float64(s.samples))
	pm.DecodeStarvedFraction.Set(s.decode.starvedFrac)
	pm.DecodeStarved.Set(boolGauge(s.decode.starved))
	pm.DecodeUnderutilizedFraction.Set(s.under.idleFrac)
	pm.DecodeUnderutilized.Set(boolGauge(s.under.under))
}

func boolGauge(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// ObserveCharge records a leg's estimated compute tokens.
func (m *Manager) ObserveCharge(tokens int64) { m.m.JobTokens.Observe(float64(tokens)) }

// UnknownCostPer1K is Config.UnknownCostPer1K.
func (m *Manager) UnknownCostPer1K() time.Duration { return m.cfg.UnknownCostPer1K }

// ActivePrefill keeps the members of cands that take prefill legs: prefill
// nodes not returning to decode.
func (m *Manager) ActivePrefill(cands []*registry.Backend) []*registry.Backend {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*registry.Backend, 0, len(cands))
	for _, b := range cands {
		if n := m.nodeLocked(b); n.mode == Prefill && !n.releasing {
			out = append(out, b)
		}
	}
	return out
}

// estLocked is a node's estimated time for one more leg, in seconds: its
// recent service time, or the fleet's when it has none, grown by the legs it
// is already running.
func (m *Manager) estLocked(n *node, now time.Time) float64 {
	svc := n.svc
	if svc == 0 || now.Sub(n.svcAt) > svcStale {
		svc = m.svcFleet
	}
	return svc * (1 + float64(n.legs)/legParallelism)
}

// PickPrefill is the active prefill node a leg goes to: the lowest est plus
// UnknownCostPer1K per 1,000 of the prompt's total tokens the node does not
// hold (depths[i] is what active[i] holds). Ties go to fewer legs in flight,
// then the lower URL. Nil when active is empty.
func (m *Manager) PickPrefill(active []*registry.Backend, total int64, depths []int64) *registry.Backend {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.cfg.Clock.Now()
	perToken := m.cfg.UnknownCostPer1K.Seconds() / 1000
	var best *registry.Backend
	var bestCost float64
	var bestLegs int
	for i, b := range active {
		n := m.nodeLocked(b)
		var held int64
		if i < len(depths) {
			held = depths[i]
		}
		cost := m.estLocked(n, now) + float64(max(total-held, 0))*perToken
		switch {
		case best == nil, cost < bestCost,
			cost == bestCost && n.legs < bestLegs,
			cost == bestCost && n.legs == bestLegs && strings.Compare(b.URL, best.URL) < 0:
			best, bestCost, bestLegs = b, cost, n.legs
		}
	}
	return best
}

// LegStart counts a P leg sent to b.
func (m *Manager) LegStart(b *registry.Backend) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nodeLocked(b).legs++
	m.publishEstLocked(m.nodes[b.URL])
}

// LegDone counts a P leg on b finished after svc, ok when it returned its
// transfer params. A node returning to decode goes once its last leg is done.
func (m *Manager) LegDone(b *registry.Backend, svc time.Duration, ok bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := m.nodes[b.URL]
	if n == nil {
		return
	}
	n.legs = max(n.legs-1, 0)
	if ok {
		now := m.cfg.Clock.Now()
		s := svc.Seconds()
		if n.svc == 0 || now.Sub(n.svcAt) > svcStale {
			n.svc = s
		} else {
			n.svc += svcAlpha * (s - n.svc)
		}
		n.svcAt = now
		if m.svcFleet == 0 {
			m.svcFleet = s
		} else {
			m.svcFleet += fleetAlpha * (s - m.svcFleet)
		}
	}
	if n.releasing && n.legs == 0 {
		m.finishReleaseLocked(n)
	}
	m.publishEstLocked(n)
}

func (m *Manager) finishReleaseLocked(n *node) {
	reason := n.releaseReason
	if reason == "" {
		reason = "ttft_low"
	}
	n.releasing, n.releaseReason = false, ""
	m.releaseForLocked(n, reason)
	m.actionLocked("release", reason)
}

func (m *Manager) actionLocked(action, reason string) {
	if m.cfg.PDMetrics != nil {
		m.cfg.PDMetrics.Action(action, reason)
	}
}

func (m *Manager) publishEstLocked(n *node) {
	if m.cfg.PDMetrics != nil && n != nil {
		m.cfg.PDMetrics.SetNodeEst(n.b.URL, m.estLocked(n, m.cfg.Clock.Now()))
	}
}

// tickPDLocked is Tick under PD.
func (m *Manager) tickPDLocked(cands []*registry.Backend) {
	now := m.cfg.Clock.Now()
	for _, b := range cands {
		m.nodeLocked(b)
	}
	for _, n := range m.nodes {
		switch {
		case n.mode == Draining && n.b.Inflight() == 0:
			m.promoteLocked(n, now)
		case n.releasing && n.legs == 0:
			m.finishReleaseLocked(n)
		}
	}
	var decode []*registry.Backend
	nonDecode := 0
	for _, b := range cands {
		if m.nodes[b.URL].mode == Decode {
			decode = append(decode, b)
		} else {
			nonDecode++
		}
	}
	snap := m.snapshotLocked(now)
	eligible := m.drainCandidatesLocked(decode)
	if m.cfg.PDMetrics != nil {
		m.cfg.PDMetrics.DrainCandidates.Set(float64(len(eligible)))
	}
	if nonDecode < m.cfg.MinPrefill && len(decode) > 0 && m.canConvertLocked(len(cands), nonDecode, len(decode)) {
		// Below the minimum (a prefill node gone unhealthy, or none named
		// initially): drain toward it, one node per tick.
		if len(eligible) > 0 {
			m.drainLocked(eligible, "min_nodes")
			m.actionLocked("drain", "min_nodes")
		} else if m.lastAction.IsZero() || now.Sub(m.lastAction) >= max(snap.span, m.cfg.DrainMinInterval) {
			m.blockedLocked(now, "min_nodes")
		}
	} else {
		m.provisionLocked(now, cands, decode, nonDecode, snap)
	}
	m.publishSnapshotLocked(snap)
	for _, n := range m.nodes {
		if n.mode == Prefill {
			m.publishEstLocked(n)
		}
	}
	m.publishCountsLocked()
}

// provisionLocked takes at most one provisioning action.
func (m *Manager) provisionLocked(now time.Time, cands, decode []*registry.Backend, nonDecode int, snap snapshot) {
	mean, k := snap.mean, snap.samples
	if k == 0 {
		return // no recent request: nothing to act on, either way
	}
	if span := snap.span; span > 0 {
		// Time mode: the window restarted at the last action, so the next
		// one waits for a full window of samples taken after it, and for
		// DrainMinInterval. A drain changes the prefill-finish time only once
		// it has become a prefill node (10-35s), so a shorter look would act
		// again on the backlog the first action already answered.
		if !m.lastAction.IsZero() && now.Sub(m.lastAction) < max(span, m.cfg.DrainMinInterval) {
			return
		}
		if k < minWindowSamples {
			return // too few requests to act on either way
		}
	} else {
		if !m.lastAction.IsZero() && now.Sub(m.lastAction) < m.cfg.DrainMinInterval {
			return
		}
		if m.ttft.since < m.cfg.TTFTWindow {
			return // the window has not refreshed since the last action
		}
	}
	starved := snap.decode.starved
	attrs := []any{"prefill_finish_mean", mean.Round(time.Millisecond), "prefill_finish_samples", k,
		"window", snap.span, "decode_starved", starved, "decode_starved_fraction", snap.decode.starvedFrac,
		"decode_underutilized", snap.under.under, "decode_idle_fraction", snap.under.idleFrac,
		"decode_samples", snap.decode.n}
	grow := func(reason string) {
		if len(decode) == 0 || !m.canConvertLocked(len(cands), nonDecode, len(decode)) {
			return
		}
		eligible := m.drainCandidatesLocked(decode)
		if len(eligible) == 0 {
			m.blockedLocked(now, reason, attrs...)
			return
		}
		m.drainLocked(eligible, reason, attrs...)
		m.actedLocked(now, "drain", reason)
	}
	if starved {
		// The decode side is out of room: give it prefill nodes back unless
		// prefill is very slow, and add one only then.
		switch {
		case mean >= m.cfg.StarvedTTFTHigh:
			grow("decode_starved_ttft_high")
		case mean < m.cfg.StarvedShrinkGuard:
			m.shrinkLocked(now, "decode_starved_shrink", "decode_starved_shrink", attrs)
		default:
			// A decision too: it takes the window, so it is logged and
			// counted once per window rather than at every tick.
			if m.cfg.Log != nil {
				m.cfg.Log.Info("prefill set holds: decode starved", append([]any{"reason", "decode_starved_hold"}, attrs...)...)
			}
			m.actedLocked(now, "hold", "decode_starved_hold")
		}
		return
	}
	if snap.under.under {
		// Decode has room to spare: never give it prefill nodes back, add
		// one if prefill is slow, else hold.
		if mean >= m.cfg.UnderutilTTFTHigh {
			grow("underutil_ttft_high")
			return
		}
		if m.cfg.Log != nil {
			m.cfg.Log.Info("prefill set holds: decode underutilized", append([]any{"reason", "underutil_hold"}, attrs...)...)
		}
		m.actedLocked(now, "hold", "underutil_hold")
		return
	}
	switch {
	case mean >= m.cfg.TTFTHigh:
		grow("ttft_high")
	case mean <= m.cfg.TTFTLow:
		m.shrinkLocked(now, "ttft_low", "ttft_low_abandon", attrs)
	}
}

// shrinkLocked takes one prefill node away: the newest drain still in
// progress, abandoned, or else the active prefill node holding the fewest
// prefix tokens, which takes no new leg and becomes a decode node when its
// legs end. Never below MinPrefill.
func (m *Manager) shrinkLocked(now time.Time, reason, abandonReason string, attrs []any) {
	if d := m.newestDrainLocked(); d != nil {
		m.releaseForLocked(d, abandonReason)
		m.actedLocked(now, "abandon", abandonReason)
		return
	}
	if m.releasingLocked() || m.activeLocked() <= m.cfg.MinPrefill {
		return
	}
	v := m.releaseCandidateLocked()
	if v == nil {
		return
	}
	v.releasing, v.releaseReason = true, reason
	m.logLocked("prefill node returning to decode", v, reason, append(attrs, "legs_in_flight", v.legs)...)
	m.restartWindowLocked(now)
	if v.legs == 0 {
		m.finishReleaseLocked(v)
	}
}

func (m *Manager) actedLocked(now time.Time, action, reason string) {
	m.restartWindowLocked(now)
	m.actionLocked(action, reason)
}

// restartWindowLocked marks an action: the window counts afresh from now.
// drainCandidatesLocked is the decode nodes a drain may pick: under
// DrainSkipStarved, those whose latest sample was not starved.
func (m *Manager) drainCandidatesLocked(decode []*registry.Backend) []*registry.Backend {
	if !m.cfg.DrainSkipStarved {
		return decode
	}
	var out []*registry.Backend
	for _, b := range decode {
		if !m.nodes[b.URL].starved {
			out = append(out, b)
		}
	}
	return out
}

// blockedLocked is a drain wanted for reason with every decode node starved:
// none is taken from where room is short; the window holds and restarts.
func (m *Manager) blockedLocked(now time.Time, reason string, attrs ...any) {
	if m.cfg.Log != nil {
		m.cfg.Log.Info("prefill set holds: every decode node starved, none drained",
			append([]any{"reason", "drain_blocked_all_starved", "wanted", reason}, attrs...)...)
	}
	m.actedLocked(now, "hold", "drain_blocked_all_starved")
}

func (m *Manager) restartWindowLocked(now time.Time) {
	m.lastAction, m.ttft.since = now, 0
	m.ttft.timed = m.ttft.timed[:0]
	m.starve = m.starve[:0]
}

func (m *Manager) newestDrainLocked() *node {
	var newest *node
	for _, n := range m.nodes {
		if n.mode == Draining && (newest == nil || n.convertedAt.After(newest.convertedAt)) {
			newest = n
		}
	}
	return newest
}

func (m *Manager) releasingLocked() bool {
	for _, n := range m.nodes {
		if n.releasing {
			return true
		}
	}
	return false
}

// activeLocked counts the healthy prefill nodes not returning to decode.
func (m *Manager) activeLocked() (k int) {
	for _, n := range m.nodes {
		if n.mode == Prefill && !n.releasing && n.b.Available() {
			k++
		}
	}
	return k
}

// releaseCandidateLocked is the active prefill node that loses least by
// returning to decode: the fewest prefix tokens held in the prefill tree,
// then the fewest legs in flight, then the newest.
func (m *Manager) releaseCandidateLocked() *node {
	var held map[string]int64
	if m.cfg.PrefixValues != nil {
		held = m.cfg.PrefixValues()
	}
	var best *node
	for _, n := range m.nodes {
		if n.mode != Prefill || n.releasing || !n.b.Available() {
			continue
		}
		switch {
		case best == nil,
			held[n.b.URL] < held[best.b.URL],
			held[n.b.URL] == held[best.b.URL] && n.legs < best.legs,
			held[n.b.URL] == held[best.b.URL] && n.legs == best.legs && n.convertedAt.After(best.convertedAt):
			best = n
		}
	}
	return best
}
