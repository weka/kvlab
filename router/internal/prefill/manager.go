// Package prefill moves heavy prefills off decode nodes.
//
// In the common case every node serves prefill and decode. When the engine
// refuses a request because recomputing its prompt would stall the node's
// running decodes (HTTP 503, header X-Weka-Prefill-Reject), the router runs that
// prefill on a node it has taken out of decode routing, then sends the request
// for decode as usual. The prefill leaves its KV in the fleet's shared tier, so
// the decode node loads it rather than recomputing.
//
// A node is in one of three modes. It starts in decode, becomes draining when a
// rejected request needs a prefill node and none has room, becomes prefill once
// its running decode requests have finished, and returns to decode once it has
// no job running or queued (after Config.ReleaseIdle, when set). Which nodes are
// prefill nodes is therefore the router's decision, made from traffic, with
// nothing configured per node.
//
// The same manager runs dynamic vanilla P/D (Config.PD): there every request's
// prefill leg is a job, handed off to the decode node over NIXL (see pdmode.go).
package prefill

import (
	"context"
	"errors"
	"log/slog"
	"math/rand/v2"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/weka/kvlab/router/internal/clock"
	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/registry"
)

// Mode is a node's role under the split.
type Mode int

const (
	// Decode is the default: the node takes ordinary traffic.
	Decode Mode = iota
	// Draining takes no new decode routing and waits for its running decode
	// requests to finish. Prefill jobs may already be queued on it.
	Draining
	// Prefill runs prefill jobs only.
	Prefill
)

func (m Mode) String() string { return metrics.PrefillModes[m] }

const (
	DefaultNodeBudgetTokens = 4_000_000
	DefaultMaxFleetFraction = 0.5

	// drainPoll is how often a queued job re-checks whether its node's decode
	// work has finished. Drain is bounded by the longest running decode, so
	// this is for responsiveness, not precision.
	drainPoll = 25 * time.Millisecond
)

// ErrNoPrefillNode reports that no node can take the job: none is a prefill
// node, and none can become one without leaving the fleet no decode node or
// exceeding the prefill share.
var ErrNoPrefillNode = errors.New("prefill: no prefill node available")

// Config tunes the split.
type Config struct {
	// NodeBudgetTokens bounds the prefill tokens queued plus in flight on one
	// node. A job bigger than the budget still runs on an otherwise empty node.
	NodeBudgetTokens int64
	// BudgetSoft (reject mode) makes the budget steer placement and conversion
	// only: a job on a prefill node is sent to the engine at once, whatever the
	// node's running tokens, instead of waiting in the router for room (see
	// canRunLocked).
	BudgetSoft bool
	// MaxFleetFraction caps the share of the fleet that is draining or
	// prefill. At least one node always stays in decode.
	MaxFleetFraction float64
	// NodeConcurrency is the router's hard per-node in-flight limit
	// (--max-node-concurrency). A prefill node never runs more jobs than this
	// at once, and counts the node's other in-flight requests against it; jobs
	// beyond it wait their turn. 0 is no limit.
	NodeConcurrency int64
	// ConvertAfter holds back draining another decode node until the
	// least-loaded prefill node has been over its budget for this long; jobs
	// queue on it meanwhile. A drained node takes no new decode work and no
	// prefill until its decodes finish, so converting on every brief overflow
	// costs more decode capacity than the queue it relieves. 0 converts at once.
	ConvertAfter time.Duration
	// ReleaseIdle keeps a prefill node out of decode until it has had no job
	// for this long, so a burst that arrives just after its last job finds it
	// ready instead of draining a node afresh. 0 releases at once.
	ReleaseIdle time.Duration
	// DrainMinInterval is the least time between two drain decisions. An
	// overflow from a charge that over-estimates the work, or from a burst
	// that finishes quickly, gets that long to clear before another decode
	// node is given up for it; meanwhile jobs queue on the prefill nodes. It
	// never holds back the drain that gives the pool its first prefill node.
	// 0 is no limit.
	DrainMinInterval time.Duration
	// MaxDraining (reject mode) caps how many nodes drain at once; 0 is no
	// cap.
	MaxDraining int
	// TTFTWindow, TTFTHigh and TTFTLow (PD only) drive provisioning: the
	// mean time to the prefill's finish (arrival to the P leg's
	// kv_transfer_params) of the last TTFTWindow P/D requests above TTFTHigh
	// drains a decode node into prefill, below TTFTLow returns one to decode
	// (see pdttft.go).
	TTFTWindow        int
	TTFTHigh, TTFTLow time.Duration
	// TTFTWindowTime, when set, makes the window the samples of the last
	// TTFTWindowTime instead of the last TTFTWindow (see pdttft.go).
	TTFTWindowTime time.Duration
	// The decode-starvation signal (PD only, decodestarve.go): DecodeProbe
	// reads a decode node every DecodeSampleInterval; it is starved at KV
	// usage >= StarveKV with >= StarveWaiting waiting; the decode side is
	// starved while the window's mean starved share is >= StarveFraction.
	// Then a prefill node returns to decode while the prefill-finish mean is
	// below StarvedShrinkGuard, one is added at StarvedTTFTHigh or above, and
	// in between the decision is to hold.
	DecodeProbe                         DecodeProbe
	DecodeSampleInterval                time.Duration
	StarveKV, StarveWaiting             float64
	StarveFraction                      float64
	StarvedShrinkGuard, StarvedTTFTHigh time.Duration
	// DrainSkipStarved (PD): a drain never picks a decode node whose latest
	// sample was starved, whatever the reason; with all of them starved the
	// window holds (drain_blocked_all_starved).
	DrainSkipStarved bool
	// Underutilization (PD only, decodestarve.go): a decode node is idle at
	// KV usage < IdleKV with nothing waiting; the decode side is
	// underutilized while the mean idle share over UnderutilWindowTime is
	// >= IdleFraction (> 1 disables it). Then UnderutilWindowTime is the
	// window, no prefill node is given back, and one is added at
	// UnderutilTTFTHigh or above.
	IdleKV, IdleFraction                   float64
	UnderutilTTFTHigh, UnderutilWindowTime time.Duration
	// UnknownCostPer1K (PD only) is what a P leg's routing cost adds per 1,000
	// prompt tokens the candidate node is not known to hold; tokens it holds
	// are free.
	UnknownCostPer1K time.Duration
	// PrefixValues (PD only) is how many tokens each backend's prefill prefix
	// tree holds, by URL: the prefill node returning to decode is the one
	// holding least.
	PrefixValues func() map[string]int64
	// PDMetrics (PD only) receives the provisioning metrics.
	PDMetrics *metrics.PDMetrics

	// StoreUnitTokens is the granularity the shared KV tier stores a prompt at:
	// a finished job leaves floor(prompt / unit) * unit tokens loadable by any
	// node. 0 means the whole prompt.
	StoreUnitTokens int64
	// CostPick lets the Picker choose among ALL eligible prefill nodes, instead
	// of only breaking ties among the fullest. Set under cost routing, where the
	// picker is the cost choice over the prefill flow's own tree; unset, jobs
	// pack onto the fullest node (the ladder). Eligibility, conversion and
	// release are the manager's either way.
	CostPick bool
	// OnModeChange, when set, is told of every role change of a node, with the
	// manager's lock held: it must be quick and must not call back into the
	// manager.
	OnModeChange func(b *registry.Backend, from, to Mode)

	// PD is dynamic vanilla P/D: every request's prefill leg is a job, and a
	// node out of decode mode never takes a decode leg (DecodeNodes has no
	// fallback). Initial and MinPrefill apply only with it set.
	PD bool
	// Initial names nodes, by URL, that are prefill from the start. They are
	// released like any other; MinPrefill is what keeps nodes in prefill.
	Initial []string
	// MinPrefill is the fewest prefill nodes: none is released below it, and
	// decode nodes are drained to reach it.
	MinPrefill int
	// Log, when set, gets a line for every conversion and release with the
	// load behind it.
	Log *slog.Logger

	Clock    clock.Clock
	PoolName string
}

func (c Config) withDefaults() Config {
	if c.NodeBudgetTokens <= 0 {
		c.NodeBudgetTokens = DefaultNodeBudgetTokens
	}
	if c.MaxFleetFraction <= 0 || c.MaxFleetFraction > 1 {
		c.MaxFleetFraction = DefaultMaxFleetFraction
	}
	if c.Clock == nil {
		c.Clock = clock.Real{}
	}
	return c
}

// Picker chooses among prefill nodes with room (see Config.CostPick). The
// gateway passes the pool's routing flow, so the choice uses the same prefix
// knowledge and load rules as ordinary routing. queued is a node's pending job
// tokens, queued and running: the prefill work ahead of a new job there. It
// reads the manager's state under the lock the picker is called with, so it is
// valid only during the call.
type Picker func(eligible []*registry.Backend, queued func(*registry.Backend) int64) (*registry.Backend, error)

type node struct {
	b *registry.Backend
	// running is the jobs holding a slot and runningTokens their tokens;
	// queue the jobs waiting for one, in arrival order.
	running       int
	runningTokens int64
	queue         []*Job
	mode          Mode
	// pending is the tokens of every job on the node, running or queued.
	pending int64
	// overSince is when pending last rose above the budget (zero while
	// within it); idleSince when a prefill node last ran out of jobs (zero
	// while it has any).
	overSince, idleSince time.Time
	// convertedAt is when it last left decode; zero for an initial node, so
	// initial nodes count as the oldest.
	convertedAt time.Time
	// Under PD (pdttft.go): releasing marks the prefill node returning to
	// decode, which takes no new leg; legs is its P legs in flight; svc an
	// EWMA of their service time in seconds, last updated at svcAt.
	releasing bool
	// releaseReason is why a releasing node returns to decode.
	releaseReason string
	// starved is the node's latest decode sample (PD): KV usage >= StarveKV
	// and waiting >= StarveWaiting. No sample, or an unread one, is false.
	starved bool
	legs    int
	svc     float64
	svcAt   time.Time
}

// Manager owns the node modes for one pool.
type Manager struct {
	cfg Config
	m   *metrics.PrefillMetrics

	mu    sync.Mutex
	nodes map[string]*node
	// lastDrain is when a decode node was last drained (DrainMinInterval).
	lastDrain time.Time
	// initial is Config.Initial, normalised, less the nodes already met: a
	// node is made a prefill node by it once, when first seen.
	initial map[string]bool
	// ttft is the provisioning window, lastAction the time of the last
	// provisioning action and svcFleet the fleet's P-leg service EWMA
	// (pdttft.go).
	ttft       ttftWindow
	starve     []starveSample
	lastAction time.Time
	svcFleet   float64

	// Mirrors of the Prometheus counters, which cannot be read back, for the
	// /router-viz state.
	rejections, jobRetries, jobsOK, jobsFailed, conversions, releases, fallbackNoNode, fallbackJobFailed, jobMoves, jobsCancelled, fallbackSkipped, drainsDeferred atomic.Uint64
}

// New builds a manager for one pool.
func New(cfg Config) *Manager {
	cfg = cfg.withDefaults()
	m := &Manager{cfg: cfg, m: metrics.ForPrefill(cfg.PoolName), nodes: map[string]*node{}, initial: map[string]bool{}}
	if cfg.PD {
		for _, u := range cfg.Initial {
			m.initial[normalize(u)] = true
		}
	}
	return m
}

// Job is a charge against a prefill node, released with Done.
type Job struct {
	mgr    *Manager
	n      *node
	tokens int64
	once   sync.Once
	// granted closes when the job holds a slot; held is set with it.
	granted chan struct{}
	held    bool
	queued  bool
	// reservedAt is when Reserve placed it: its router wait runs from here.
	reservedAt time.Time
	// Backend is the node the job runs on.
	Backend *registry.Backend
}

// Done releases the job's tokens and stamps the node as recently busy.
func (j *Job) Done() {
	j.once.Do(func() {
		m := j.mgr
		m.mu.Lock()
		defer m.mu.Unlock()
		m.dequeueLocked(j)
		m.unholdLocked(j)
		j.n.pending -= j.tokens
		// A node pruned while the job ran is no longer the manager's: its
		// counters settle, but its mode, metrics and release belong to
		// whichever node now has the URL.
		if m.nodes[j.n.b.URL] != j.n {
			return
		}
		m.trackBudgetLocked(j.n)
		m.releaseIfIdleLocked(j.n)
		m.publishLocked(j.n)
		m.spreadLocked()
	})
}

// unholdLocked returns a held job's slot and tokens to its node and hands them
// on.
func (m *Manager) unholdLocked(j *Job) {
	if !j.held {
		return
	}
	j.held = false
	j.n.running--
	j.n.runningTokens -= j.tokens
	m.grantLocked(j.n)
}

func (m *Manager) nodeLocked(b *registry.Backend) *node {
	n := m.nodes[b.URL]
	if n == nil {
		n = &node{b: b}
		if key := normalize(b.URL); m.initial[key] {
			delete(m.initial, key)
			n.mode = Prefill
		}
		m.nodes[b.URL] = n
		if n.mode == Prefill {
			m.publishLocked(n)
		}
	}
	n.b = b
	return n
}

// ModeOf reports a backend's current mode.
func (m *Manager) ModeOf(b *registry.Backend) Mode {
	m.mu.Lock()
	defer m.mu.Unlock()
	if n := m.nodes[b.URL]; n != nil {
		return n.mode
	}
	return Decode
}

// DecodeNodes is the candidate set for ordinary routing: everything not
// draining or prefill. When that would leave nothing — the decode nodes have
// all gone unhealthy — it returns cands unchanged, since a request served on a
// prefill node beats one refused.
func (m *Manager) DecodeNodes(cands []*registry.Backend) []*registry.Backend {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*registry.Backend, 0, len(cands))
	for _, b := range cands {
		if !m.inPrefillLocked(b) {
			out = append(out, b)
		}
	}
	if len(out) == 0 && !m.cfg.PD {
		return cands
	}
	return out
}

// StoreCut is how many leading tokens of a prompt of the given size a finished
// job leaves in the shared tier.
func (m *Manager) StoreCut(tokens int64) int64 {
	if u := m.cfg.StoreUnitTokens; u > 0 {
		return tokens / u * u
	}
	return tokens
}

// Reserve picks the node a prefill job of tokens runs on and charges it.
//
//  1. Among prefill nodes with budget room, take the one with the most pending
//     tokens; the routing policy only breaks ties (with CostPick it chooses
//     among them all). Draining nodes take no job while a prefill node exists:
//     a job there would wait for the node's decodes to finish while a prefill
//     node could run it. With no prefill node yet the draining nodes stand in.
//  2. Failing that, while the prefill share is under the cap, drain the
//     least-loaded decode node: with ConvertAfter set only once the
//     least-loaded prefill node has been over budget that long, no sooner
//     than DrainMinInterval after the last drain, with fewer than MaxDraining
//     nodes draining, and only while the pending tokens exceed the budget of
//     the prefill and draining nodes together (a draining node is a budget of
//     room on its way). The job goes to the drained
//     node when it has nothing in flight
//     (its WaitReady makes it a prefill node at once) or there is no prefill
//     node; otherwise it queues on the least-pending prefill node.
//  3. Failing that, queue on the prefill node (or, with none, the draining
//     node) with the least pending tokens.
//  4. With neither, ErrNoPrefillNode.
//
// cands is the pool's healthy backends: the fleet the cap is a share of.
func (m *Manager) Reserve(cands []*registry.Backend, tokens int64, pick Picker) (*Job, error) {
	tokens = max(tokens, 1)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.m.JobTokens.Observe(float64(tokens))

	var decode, prefillNodes, draining []*registry.Backend
	for _, b := range cands {
		switch m.nodeLocked(b).mode {
		case Decode:
			decode = append(decode, b)
		case Draining:
			draining = append(draining, b)
		default:
			prefillNodes = append(prefillNodes, b)
		}
	}
	// The nodes that take jobs: prefill nodes, or the draining ones while
	// there is no prefill node yet.
	takers := prefillNodes
	if len(takers) == 0 {
		takers = draining
	}
	var eligible []*registry.Backend
	for _, b := range takers {
		if n := m.nodes[b.URL]; n.pending == 0 || n.pending+tokens <= m.cfg.NodeBudgetTokens {
			eligible = append(eligible, b)
		}
	}
	nonDecode := len(prefillNodes) + len(draining)

	var chosen *registry.Backend
	switch {
	case len(eligible) > 0:
		chosen = m.pickLocked(eligible, pick)
	case m.canConvertLocked(len(cands), nonDecode, len(decode)) &&
		m.overflowedLongEnoughLocked(takers) && m.drainNeededLocked(prefillNodes, draining, tokens) &&
		m.drainAllowedLocked(nonDecode, len(draining)):
		d := m.drainLocked(decode, "no_prefill_room", "job_tokens", tokens)
		chosen = d.b
		// A drained node still decoding would hold the job until it is done;
		// one with nothing in flight runs it at once.
		if d.b.Inflight() > 0 && len(prefillNodes) > 0 {
			chosen = m.leastPendingLocked(prefillNodes)
		}
	case len(takers) > 0:
		chosen = m.leastPendingLocked(takers)
	default:
		return nil, ErrNoPrefillNode
	}

	n := m.nodes[chosen.URL]
	n.pending += tokens
	n.idleSince = time.Time{}
	m.trackBudgetLocked(n)
	m.publishLocked(n)
	j := &Job{mgr: m, n: n, tokens: tokens, Backend: chosen, granted: make(chan struct{}), queued: true,
		reservedAt: m.cfg.Clock.Now()}
	n.queue = append(n.queue, j)
	m.publishQueuedLocked()
	m.spreadLocked()
	return j, nil
}

// drainLocked starts draining the least-loaded of decode, for reason, and
// logs it with attrs.
func (m *Manager) drainLocked(decode []*registry.Backend, reason string, attrs ...any) *node {
	target := leastLoaded(decode)
	if m.cfg.PD {
		target = soonestDrained(decode)
	}
	n := m.nodes[target.URL]
	n.starved = false // its decode samples no longer apply
	m.setModeLocked(n, Draining)
	now := m.cfg.Clock.Now()
	n.convertedAt, m.lastDrain = now, now
	m.conversions.Add(1)
	m.m.Conversions.Inc()
	m.publishLocked(n)
	m.logLocked("decode node draining for prefill", n, reason, attrs...)
	return n
}

func (m *Manager) pickLocked(eligible []*registry.Backend, pick Picker) *registry.Backend {
	if m.cfg.CostPick && pick != nil {
		// A policy that declines (every node saturated by some signal) must not
		// refuse work the budget says fits: fall through to packing.
		if b, err := pick(eligible, m.pendingLocked); err == nil && b != nil {
			return b
		}
	}
	var fullest []*registry.Backend
	for _, b := range eligible {
		switch p := m.nodes[b.URL].pending; {
		case len(fullest) == 0 || p > m.nodes[fullest[0].URL].pending:
			fullest = append(fullest[:0], b)
		case p == m.nodes[fullest[0].URL].pending:
			fullest = append(fullest, b)
		}
	}
	if len(fullest) > 1 && pick != nil {
		// The policy declining — a saturation signal on a node already
		// running prefill jobs is expected — must not refuse work the budget
		// says fits, so any tied node will do.
		if b, err := pick(fullest, m.pendingLocked); err == nil && b != nil {
			return b
		}
	}
	return fullest[0]
}

// pendingLocked is b's pending job tokens, 0 for a node the manager has not
// seen.
func (m *Manager) pendingLocked(b *registry.Backend) int64 {
	if n := m.nodes[b.URL]; n != nil {
		return n.pending
	}
	return 0
}

// drainNeededLocked is whether the work waiting for prefill, this job
// included, is more than the prefill nodes and the nodes already draining can
// take within their budgets: each draining node counts as a budget of room,
// so a sustained overflow drains toward its size instead of one node per
// DrainMinInterval for as long as it lasts.
func (m *Manager) drainNeededLocked(prefillNodes, draining []*registry.Backend, tokens int64) bool {
	pending := tokens
	for _, b := range prefillNodes {
		pending += m.nodes[b.URL].pending
	}
	for _, b := range draining {
		pending += m.nodes[b.URL].pending
	}
	return pending > m.cfg.NodeBudgetTokens*int64(len(prefillNodes)+len(draining))
}

// drainAllowedLocked applies MaxDraining and DrainMinInterval to a drain that
// is otherwise due, counting the one it defers.
// counting the one it defers. With no prefill node at all the drain goes
// ahead: there is nowhere for the job to queue.
func (m *Manager) drainAllowedLocked(nonDecode, draining int) bool {
	if m.cfg.MaxDraining > 0 && draining >= m.cfg.MaxDraining {
		m.drainsDeferred.Add(1)
		m.m.DrainsDeferred.Inc()
		return false
	}
	if m.cfg.DrainMinInterval <= 0 || nonDecode == 0 || m.lastDrain.IsZero() ||
		m.cfg.Clock.Now().Sub(m.lastDrain) >= m.cfg.DrainMinInterval {
		return true
	}
	m.drainsDeferred.Add(1)
	m.m.DrainsDeferred.Inc()
	return false
}

func (m *Manager) leastPendingLocked(bs []*registry.Backend) *registry.Backend {
	best, ties := bs[0], 1
	for _, b := range bs[1:] {
		switch p, bp := m.nodes[b.URL].pending, m.nodes[best.URL].pending; {
		case p < bp:
			best, ties = b, 1
		case p == bp:
			ties++
			if rand.IntN(ties) == 0 {
				best = b
			}
		}
	}
	return best
}

func leastLoaded(bs []*registry.Backend) *registry.Backend {
	best, ties := bs[0], 1
	for _, b := range bs[1:] {
		switch l, bl := b.NormalizedLoad(), best.NormalizedLoad(); {
		case l < bl:
			best, ties = b, 1
		case l == bl:
			ties++
			if rand.IntN(ties) == 0 {
				best = b
			}
		}
	}
	return best
}

// soonestDrained is the decode node to drain under PD: the one whose oldest
// in-flight request started last (a node with nothing in flight first), so
// the wait for its decodes is shortest; fewest in-flight requests breaks a tie.
func soonestDrained(bs []*registry.Backend) *registry.Backend {
	best := bs[0]
	bestOld, bestN := best.OldestStream(), best.Inflight()
	for _, b := range bs[1:] {
		old, n := b.OldestStream(), b.Inflight()
		switch {
		case bestN == 0 && n != 0:
		case n == 0 && bestN != 0, old.After(bestOld), old.Equal(bestOld) && n < bestN:
			best, bestOld, bestN = b, old, n
		}
	}
	return best
}

// canConvertLocked applies the cap: floor(fleet * fraction) nodes may be
// draining or prefill, and the last decode node is never converted.
func (m *Manager) canConvertLocked(fleet, nonDecode, decode int) bool {
	limit := min(int(float64(fleet)*m.cfg.MaxFleetFraction), fleet-1)
	return decode > 1 && nonDecode < limit
}

// overflowedLongEnoughLocked applies ConvertAfter: with no prefill node yet,
// or the least-loaded one over budget for at least that long, a decode node
// may be drained.
func (m *Manager) overflowedLongEnoughLocked(nonDecode []*registry.Backend) bool {
	if m.cfg.ConvertAfter <= 0 || len(nonDecode) == 0 {
		return true
	}
	n := m.nodes[m.leastPendingLocked(nonDecode).URL]
	return !n.overSince.IsZero() && m.cfg.Clock.Now().Sub(n.overSince) >= m.cfg.ConvertAfter
}

func (m *Manager) trackBudgetLocked(n *node) {
	switch {
	case n.pending <= m.cfg.NodeBudgetTokens:
		n.overSince = time.Time{}
	case n.overSince.IsZero():
		n.overSince = m.cfg.Clock.Now()
	}
}

// WaitReady blocks until the job may run on its node; a job is queued from
// Reserve, so call it straight away. A draining node must
// first finish its in-flight requests — all decode, since no prefill job has
// been dispatched to it yet — and then becomes a prefill node. The job then
// takes a slot under the node's concurrency limit, queueing in arrival order
// behind the jobs ahead of it. The caller releases the slot with Done.
func (m *Manager) WaitReady(ctx context.Context, j *Job) error {
	for {
		m.mu.Lock()
		if j.n.mode == Draining && j.Backend.Inflight() == 0 {
			m.setModeLocked(j.n, Prefill)
			m.publishLocked(j.n)
			m.spreadLocked()
		}
		if j.n.mode != Draining {
			break
		}
		m.mu.Unlock()
		select {
		case <-j.granted: // moved to a node that was ready
		case <-m.cfg.Clock.After(drainPoll):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	// m.mu is held. The job joined its node's queue when it was reserved, so
	// the queue is in arrival order whatever the drain wait did to timing.
	for {
		m.grantLocked(j.n)
		if j.held {
			// Its node's queue may now be empty, which lets it take waiting
			// work from elsewhere.
			m.spreadLocked()
			m.mu.Unlock()
			return nil
		}
		m.mu.Unlock()
		select {
		case <-j.granted:
		case <-m.cfg.Clock.After(drainPoll):
		case <-ctx.Done():
			m.mu.Lock()
			m.dequeueLocked(j)
			// Granted in the same instant: hand the slot on.
			m.unholdLocked(j)
			m.mu.Unlock()
			return ctx.Err()
		}
		m.mu.Lock()
	}
}

// grantLocked hands free slots to the head of the node's queue. Slots are the
// node's limit less its running jobs, and the node's own in-flight count — the
// one routing reads — must be under the limit too, so decode work still on a
// node counts against the same ceiling. The poll in WaitReady re-checks the
// latter, which changes without anyone telling the manager.
func (m *Manager) grantLocked(n *node) {
	for len(n.queue) > 0 && m.canRunLocked(n, n.queue[0]) {
		j := n.queue[0]
		n.queue = n.queue[1:]
		j.queued, j.held = false, true
		n.running++
		n.runningTokens += j.tokens
		m.noteWaitLocked(j)
		close(j.granted)
		m.publishQueuedLocked()
	}
}

// canRunLocked reports whether j may start on n now: n is not draining, has a
// slot under the concurrency limit, and — with a hard budget — its running
// jobs leave room for j in the token budget. A job bigger than the budget runs
// alone.
//
// With a hard budget (the default) the budget bounds what is SENT to a node,
// not only what is placed on it. Jobs placed over it — queued on the
// least-pending node while ConvertAfter holds back a conversion — wait in the
// manager, where spreadLocked can still move them to a node that frees room or
// finishes converting. Sent to the engine they would be bound to that node
// until it ran them all.
//
// With BudgetSoft the budget bounds only placement and conversion (Reserve,
// trackBudgetLocked, overLocked): a job starts as soon as its node is a
// prefill node with a free concurrency slot, and the engine's own scheduler
// queues what the node cannot run at once. Running tokens may then exceed the
// budget. The trade is the reverse of the hard budget's: no router wait in
// front of an engine with spare capacity, but a job sent over budget stays on
// its node even when a conversion later frees room elsewhere.
func (m *Manager) canRunLocked(n *node, j *Job) bool {
	if n.mode == Draining {
		return false
	}
	if limit := m.cfg.NodeConcurrency; limit > 0 && (int64(n.running) >= limit || n.b.Inflight() >= limit) {
		return false
	}
	return m.cfg.BudgetSoft || n.running == 0 || n.runningTokens+j.tokens <= m.cfg.NodeBudgetTokens
}

// spreadLocked moves jobs waiting on a node that cannot start them onto
// prefill nodes that could start them now and have no queue of their own. It takes the head of the longest queue (by
// tokens) whose head fits somewhere, so waiting work goes wherever it can run
// first, and repeats until nothing can move. The head is the job that has
// waited longest on its node, so moving it keeps arrival order as far as the
// nodes allow.
func (m *Manager) spreadLocked() {
	for m.moveOneLocked() {
	}
}

func (m *Manager) moveOneLocked() bool {
	type src struct {
		n      *node
		queued int64
	}
	var srcs []src
	for _, n := range m.nodes {
		var q int64
		for _, j := range n.queue {
			q += j.tokens
		}
		if q > 0 {
			srcs = append(srcs, src{n, q})
		}
	}
	sort.Slice(srcs, func(i, j int) bool { return srcs[i].queued > srcs[j].queued })
	for _, s := range srcs {
		head := s.n.queue[0]
		if m.canRunLocked(s.n, head) {
			continue // about to start where it is
		}
		for _, n := range m.nodes {
			if n != s.n && n.mode == Prefill && len(n.queue) == 0 && m.canRunLocked(n, head) {
				m.moveLocked(head, s.n, n)
				return true
			}
		}
	}
	return false
}

// moveLocked moves a queued job from src to dst and starts it there.
func (m *Manager) moveLocked(j *Job, src, dst *node) {
	src.queue = src.queue[1:]
	src.pending -= j.tokens
	dst.pending += j.tokens
	j.n, j.Backend = dst, dst.b
	dst.queue = append(dst.queue, j)
	dst.idleSince = time.Time{}
	m.trackBudgetLocked(src)
	m.trackBudgetLocked(dst)
	m.grantLocked(dst)
	m.releaseIfIdleLocked(src)
	m.publishLocked(src)
	m.publishLocked(dst)
	m.jobMoves.Add(1)
	m.m.JobMoves.Inc()
}

func (m *Manager) dequeueLocked(j *Job) {
	if !j.queued {
		return
	}
	for i, q := range j.n.queue {
		if q == j {
			j.n.queue = append(j.n.queue[:i:i], j.n.queue[i+1:]...)
			break
		}
	}
	j.queued = false
	m.publishQueuedLocked()
}

func (m *Manager) publishQueuedLocked() {
	q := 0
	for _, n := range m.nodes {
		q += len(n.queue)
	}
	m.m.Queued.Set(float64(q))
}

// CountJobRetry records a prefill job retried after its node answered 429.
func (m *Manager) CountJobRetry() {
	m.jobRetries.Add(1)
	m.m.JobRetries.Inc()
}

// releaseIfIdleLocked returns a node to decode once it has no job in flight
// and none queued: at once, or for a prefill node with ReleaseIdle set, after
// that long idle (see releaseExpired). A draining node goes at once once the
// prefill nodes are no longer over (overLocked): it serves neither decode nor
// prefill while it waits. It runs under the same
// lock as every queue change, so a job reserved concurrently either finds the
// node still in prefill and keeps it, or finds it back in decode and converts
// it afresh.
func (m *Manager) releaseIfIdleLocked(n *node) {
	if n.mode == Decode || n.pending != 0 || n.running != 0 || len(n.queue) != 0 {
		return
	}
	if n.mode == Prefill && !m.mayReleaseLocked() {
		return
	}
	if n.mode == Prefill && m.cfg.ReleaseIdle > 0 {
		if n.idleSince.IsZero() {
			n.idleSince = m.cfg.Clock.Now()
		}
		return
	}
	// A drain is abandoned only once the overflow that called for it has
	// cleared; until then the node keeps draining with nothing on it, and
	// becomes a prefill node when its decodes finish.
	if n.mode == Draining && (m.overLocked() || m.nonDecodeLocked() <= m.cfg.MinPrefill) {
		return
	}
	m.releaseLocked(n)
}

// setModeLocked is the only place a node's role changes.
func (m *Manager) setModeLocked(n *node, to Mode) {
	from := n.mode
	n.mode = to
	if from != to && m.cfg.OnModeChange != nil {
		m.cfg.OnModeChange(n.b, from, to)
	}
}

func (m *Manager) releaseLocked(n *node) {
	reason := "idle"
	if n.mode == Draining {
		reason = "drain_abandoned"
	}
	m.releaseForLocked(n, reason)
}

func (m *Manager) releaseForLocked(n *node, reason string) {
	m.setModeLocked(n, Decode)
	n.idleSince = time.Time{}
	m.releases.Add(1)
	m.m.Releases.Inc()
	m.publishLocked(n)
	m.logLocked("prefill node released to decode", n, reason)
}

// releaseExpired returns to decode every prefill node idle for ReleaseIdle.
func (m *Manager) releaseExpired() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.releaseExpiredLocked()
}

func (m *Manager) releaseExpiredLocked() {
	now := m.cfg.Clock.Now()
	for _, n := range m.nodes {
		if n.mode == Prefill && !n.idleSince.IsZero() && now.Sub(n.idleSince) >= m.cfg.ReleaseIdle && m.mayReleaseLocked() {
			m.releaseLocked(n)
		}
	}
}

// Prune forgets nodes that have left the pool and drops their series.
func (m *Manager) Prune(known []*registry.Backend) {
	m.mu.Lock()
	defer m.mu.Unlock()
	live := make(map[string]bool, len(known))
	for _, b := range known {
		live[b.URL] = true
	}
	for url := range m.nodes {
		if !live[url] {
			delete(m.nodes, url)
			m.m.DropNode(url)
		}
	}
	m.publishCountsLocked()
}

// pruneEvery is how often Run forgets departed nodes.
const pruneEvery = 30 * time.Second

// Run prunes departed nodes and runs Tick, until ctx is done. known supplies the pool's current
// backends.
func (m *Manager) Run(ctx context.Context, known func() []*registry.Backend) {
	t := m.cfg.Clock.NewTicker(pruneEvery)
	defer t.Stop()
	tick := m.cfg.Clock.NewTicker(tickEvery)
	defer tick.Stop()
	if m.cfg.PD && m.cfg.DecodeProbe != nil && m.cfg.DecodeSampleInterval > 0 {
		go m.runDecodeSampler(ctx, known)
	}
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C():
			m.Prune(known())
		case <-tick.C():
			m.Tick(available(known()))
		}
	}
}

func (m *Manager) publishLocked(n *node) {
	m.m.SetNode(n.b.URL, int(n.mode), n.pending)
	m.publishCountsLocked()
}

func (m *Manager) publishCountsLocked() {
	var counts [3]int
	for _, n := range m.nodes {
		counts[n.mode]++
	}
	m.m.SetModeCounts(counts)
}

// CountRejection records a large_prefill_rejected answer the router absorbed.
func (m *Manager) CountRejection() {
	m.rejections.Add(1)
	m.m.Rejections.Inc()
}

// CountJob records the outcome of a prefill job.
func (m *Manager) CountJob(ok bool) {
	if ok {
		m.jobsOK.Add(1)
		m.m.JobsOK.Inc()
		return
	}
	m.CountJobFailed("other")
}

// CountJobCancelled records a prefill job its caller cancelled, while waiting
// or running: one of metrics.PrefillJobCancelReasons. It is not a failure of
// the job or the node.
func (m *Manager) CountJobCancelled(reason string) {
	m.jobsCancelled.Add(1)
	m.m.JobCancelled(reason)
}

// CountJobFailed records a prefill job that did not succeed, and why: one of
// metrics.PrefillJobFailureReasons.
func (m *Manager) CountJobFailed(reason string) {
	m.jobsFailed.Add(1)
	m.m.JobFailed(reason)
}

// FallbackNoNode records a rejected request served with the bypass because no
// prefill node could be used.
func (m *Manager) FallbackNoNode() {
	m.fallbackNoNode.Add(1)
	m.m.FallbackNoNode.Inc()
}

// FallbackSkipped records a request whose bypass fallback was not sent because
// its caller had already left.
func (m *Manager) FallbackSkipped() {
	m.fallbackSkipped.Add(1)
	m.m.FallbackSkipped.Inc()
}

// FallbackJobFailed records a request sent for decode with the bypass because
// its prefill job did not succeed.
func (m *Manager) FallbackJobFailed() {
	m.fallbackJobFailed.Add(1)
	m.m.FallbackJobFail.Inc()
}

// NodeState is one node's view for /router-viz.
type NodeState struct {
	URL           string `json:"url"`
	Mode          string `json:"mode"`
	PendingTokens int64  `json:"pending_tokens"`
	RunningJobs   int    `json:"running_jobs"`
	QueuedJobs    int    `json:"queued_jobs"`
	// RunningTokens is the tokens of the running jobs: what the node has been
	// sent. With a hard budget it is at most the budget unless one job exceeds
	// it; with a soft one it may be any amount.
	RunningTokens int64 `json:"running_tokens"`
}

// State is the split's view for /router-viz.
type State struct {
	Enabled          bool        `json:"enabled"`
	NodeBudgetTokens int64       `json:"node_budget_tokens"`
	MaxFleetFraction float64     `json:"max_fleet_fraction"`
	Nodes            []NodeState `json:"nodes"`
	Rejections       uint64      `json:"rejections"`
	Jobs             uint64      `json:"jobs"`
	JobsFailed       uint64      `json:"jobs_failed"`
	// JobsCancelled is jobs whose caller left; they are in neither Jobs nor
	// JobsFailed.
	JobsCancelled uint64 `json:"jobs_cancelled"`
	// FallbacksSkipped is rejected requests whose fallback was not sent
	// because the caller had left; they are not in BypassFallbacks.
	FallbacksSkipped uint64 `json:"fallbacks_skipped"`
	JobRetries       uint64 `json:"job_retries"`
	JobMoves         uint64 `json:"job_moves"`
	Conversions      uint64 `json:"conversions"`
	// DrainsDeferred is jobs that found a drain due but held back by
	// DrainMinInterval.
	DrainsDeferred  uint64 `json:"drains_deferred"`
	Releases        uint64 `json:"releases"`
	BypassFallbacks uint64 `json:"bypass_fallbacks"`
}

// State snapshots the modes, queues and counters, nodes sorted by URL.
func (m *Manager) State() State {
	m.mu.Lock()
	st := State{Enabled: true, NodeBudgetTokens: m.cfg.NodeBudgetTokens, MaxFleetFraction: m.cfg.MaxFleetFraction}
	for url, n := range m.nodes {
		st.Nodes = append(st.Nodes, NodeState{URL: url, Mode: n.mode.String(), PendingTokens: n.pending, RunningJobs: n.running, QueuedJobs: len(n.queue), RunningTokens: n.runningTokens})
	}
	m.mu.Unlock()
	sort.Slice(st.Nodes, func(i, j int) bool { return st.Nodes[i].URL < st.Nodes[j].URL })
	st.Rejections = m.rejections.Load()
	st.Jobs = m.jobsOK.Load() + m.jobsFailed.Load()
	st.JobsFailed = m.jobsFailed.Load()
	st.JobsCancelled = m.jobsCancelled.Load()
	st.FallbacksSkipped = m.fallbackSkipped.Load()
	st.JobRetries = m.jobRetries.Load()
	st.JobMoves = m.jobMoves.Load()
	st.Conversions = m.conversions.Load()
	st.DrainsDeferred = m.drainsDeferred.Load()
	st.Releases = m.releases.Load()
	st.BypassFallbacks = m.fallbackNoNode.Load() + m.fallbackJobFailed.Load()
	return st
}
