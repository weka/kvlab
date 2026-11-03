// Package registry owns the set of backends and publishes immutable,
// deterministically-ordered snapshots of it.
//
// The ordering guarantee is not cosmetic. v1 returned worker lists straight out
// of a DashMap, in unspecified order that varied between calls, which alone
// degenerated round-robin into random and made every `index 0` fallback pin
// traffic to an arbitrary backend that changed per request (WRK-N1, LB-N3).
package registry

import (
	"fmt"
	"net"
	"net/url"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/weka/wekai/router/internal/circuit"
	"github.com/weka/wekai/router/internal/waitq"
)

// Kind distinguishes a leaf inference worker from a child router.
//
// Hierarchical routing is deferred (post-v2.0), but the field ships now as a
// no-op default so it stays additive rather than becoming a refactor (HIER-1).
type Kind uint8

const (
	KindWorker Kind = iota
	KindRouter
)

func (k Kind) String() string {
	if k == KindRouter {
		return "router"
	}
	return "worker"
}

// Provenance records who put a backend here. Discovery may only ever touch
// entries it owns, so a reconcile pass can never delete a statically-declared
// backend (HIER-19, WRK-7).
type Provenance uint8

const (
	ProvStatic Provenance = iota
	ProvDiscovered
)

func (p Provenance) String() string {
	if p == ProvDiscovered {
		return "discovered"
	}
	return "static"
}

// HealthModel is declared, never inferred. A backend with no health endpoint is
// a configuration (`passive`), not a special case (API-16).
type HealthModel uint8

const (
	HealthActive HealthModel = iota
	HealthPassive
)

// Health is the observed state. A new backend starts Unknown and becomes
// eligible only after its first successful check (HLT-5, SD-N3).
type Health int32

const (
	Unknown Health = iota
	Healthy
	Unhealthy
)

func (h Health) String() string {
	switch h {
	case Healthy:
		return "healthy"
	case Unhealthy:
		return "unhealthy"
	}
	return "unknown"
}

// Spec is the desired configuration of a backend. It is the input to Add and to
// discovery reconciliation; Backend is the live object.
type Spec struct {
	URL       string
	Kind      Kind
	DialectID string
	Health    HealthModel
	Prov      Provenance
	Model     string
	Locality  string
	// Capacity is the denominator of normalized load (HIER-5). For a leaf it
	// is max_inflight_per_worker; <1 is treated as 1.
	Capacity int64
}

// Backend is a routable destination.
//
// Membership is immutable once published in a Snapshot; the mutable per-backend
// state below is atomic, so a policy holding a snapshot sees a stable
// membership list with live counters — which is exactly what WRK-4 requires.
//
// Kind, Model and Locality are ALSO atomic, even though they look like static
// identity, because Add's update path (re-registration and discovery
// reconciliation) can rewrite them on a *Backend already shared across
// published snapshots and read concurrently on the request path (gateway
// filtering, admin serialization). URL, DialectID, HealthMod and Prov are
// genuinely set once at construction and never touched by Add's update path,
// so they stay plain fields.
type Backend struct {
	URL       string // canonical
	DialectID string
	HealthMod HealthModel
	Prov      Provenance

	kind     atomic.Int32
	model    atomic.Pointer[string]
	locality atomic.Pointer[string]

	CB *circuit.Breaker

	capacity atomic.Int64
	inflight atomic.Int64
	draining atomic.Bool
	health   atomic.Int32

	// requests counts leases ever acquired on this backend: one per routed
	// attempt, a retry elsewhere counting on the backend it went to. Written
	// only by internal/lease, like inflight.
	requests atomic.Uint64

	Served, Failed atomic.Uint64

	// ttftNanos is an EWMA of time-to-first-body-byte for accepted responses,
	// 0 until the first sample. Written by the proxy, read by the routing flow's
	// TTFT signal and soft band.
	ttftMu    sync.Mutex
	ttftNanos int64
	// ttftWin is the rolling sample window behind TTFTp90 (oldest first).
	ttftWin []ttftSample

	// hardTTFT is the hard-TTFT limit in nanos (0 = off), set by the routing
	// policy. hardBlocked is the 1-in-1-out gate: set when a request is admitted
	// while TTFTp90() >= hardTTFT, cleared on ANY lease release. Gating on the flag
	// rather than on TTFT alone is what prevents a deadlock: a permanently
	// excluded node would get no samples and its EWMA would never recover.
	hardTTFT    atomic.Int64
	hardBlocked atomic.Bool

	// waiters holds requests refused only by the concurrency signal on THIS
	// backend, bound here rather than pool-wide so a wait preserves the
	// prefix-cache locality the policy picked this backend for. Woken by
	// internal/lease on every release, the same single call site that is
	// already the one legitimate mutator of inflight (see AddInflight).
	//
	// A value field, not a pointer, so a zero-value Backend has a usable queue
	// with no constructor — the same reason CB is the only pointer field here
	// that needs one.
	waiters waitq.Queue
}

// ttftAlpha weights the newest sample in the TTFT EWMA.
const ttftAlpha = 0.3

type ttftSample struct {
	at time.Time // monotonic
	d  time.Duration
}

const (
	// ttftWindow is how far back TTFTp90 looks. Samples age out on their own,
	// so a node with no fresh traffic drifts back to p90==0 (self-recovery).
	ttftWindow = 30 * time.Second
	// ttftMinSamples: below this the p90 is reported as 0 so a cold or
	// low-traffic node is never flagged on a handful of samples.
	ttftMinSamples = 8
	// ttftWindowCap bounds memory under very high request rates.
	ttftWindowCap = 8192
)

// evictTTFT drops samples older than the window. Caller holds ttftMu.
func (b *Backend) evictTTFT(now time.Time) {
	i := 0
	for i < len(b.ttftWin) && now.Sub(b.ttftWin[i].at) > ttftWindow {
		i++
	}
	if i > 0 {
		b.ttftWin = append(b.ttftWin[:0], b.ttftWin[i:]...)
	}
}

// TTFTp90 is the 90th percentile of TTFT over the last 30s, or 0 with fewer
// than 8 samples in the window. This, not the mean, is what the router's TTFT
// split reads: the mean badly understates tail degradation on a loaded node.
func (b *Backend) TTFTp90() time.Duration {
	//clockexempt: rolling-window measurement
	now := time.Now()
	b.ttftMu.Lock()
	b.evictTTFT(now)
	n := len(b.ttftWin)
	if n < ttftMinSamples {
		b.ttftMu.Unlock()
		return 0
	}
	ds := make([]time.Duration, n)
	for i, s := range b.ttftWin {
		ds[i] = s.d
	}
	b.ttftMu.Unlock()
	sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
	idx := (n*9+9)/10 - 1 // ceil(0.9*n)-1
	return ds[idx]
}

// ObserveTTFT folds one time-to-first-token sample into the backend's EWMA
// (seeding it with the first sample) and its rolling p90 window.
func (b *Backend) ObserveTTFT(d time.Duration) {
	if d <= 0 {
		return
	}
	//clockexempt: rolling-window measurement
	now := time.Now()
	b.ttftMu.Lock()
	b.evictTTFT(now)
	if len(b.ttftWin) >= ttftWindowCap {
		b.ttftWin = append(b.ttftWin[:0], b.ttftWin[1:]...)
	}
	b.ttftWin = append(b.ttftWin, ttftSample{at: now, d: d})
	if b.ttftNanos == 0 {
		b.ttftNanos = int64(d)
	} else {
		b.ttftNanos = int64(ttftAlpha*float64(d) + (1-ttftAlpha)*float64(b.ttftNanos))
	}
	b.ttftMu.Unlock()
}

// SetHardTTFT configures the hard-TTFT limit used by NoteAdmit (0 = off).
func (b *Backend) SetHardTTFT(d time.Duration) { b.hardTTFT.Store(int64(d)) }

// HardBlocked reports whether the 1-in-1-out gate is closed.
func (b *Backend) HardBlocked() bool { return b.hardBlocked.Load() }

// NoteAdmit closes the hard-TTFT gate if the backend's p90 TTFT is over its limit.
// Lease-only (called from Acquire).
func (b *Backend) NoteAdmit() {
	if h := b.hardTTFT.Load(); h > 0 && int64(b.TTFTp90()) >= h {
		b.hardBlocked.Store(true)
	}
}

// NoteRelease reopens the gate: any completion frees one slot. Lease-only.
func (b *Backend) NoteRelease() { b.hardBlocked.Store(false) }

// TTFT is the current mean time-to-first-token, 0 with no samples.
func (b *Backend) TTFT() time.Duration {
	b.ttftMu.Lock()
	defer b.ttftMu.Unlock()
	return time.Duration(b.ttftNanos)
}

func (b *Backend) Kind() Kind     { return Kind(b.kind.Load()) }
func (b *Backend) SetKind(k Kind) { b.kind.Store(int32(k)) }

func (b *Backend) Model() string {
	if p := b.model.Load(); p != nil {
		return *p
	}
	return ""
}
func (b *Backend) SetModel(m string) { b.model.Store(&m) }

func (b *Backend) Locality() string {
	if p := b.locality.Load(); p != nil {
		return *p
	}
	return ""
}
func (b *Backend) SetLocality(l string) { b.locality.Store(&l) }

func (b *Backend) Inflight() int64 { return b.inflight.Load() }

// Waiters returns this backend's wait queue, for joining (the gateway's
// capacity retry path) and waking (internal/lease on release).
func (b *Backend) Waiters() *waitq.Queue { return &b.waiters }

// Queued reports how many requests are currently waiting for a slot on this
// backend.
func (b *Backend) Queued() int64 { return int64(b.waiters.Len()) }

func (b *Backend) Capacity() int64 {
	if c := b.capacity.Load(); c > 0 {
		return c
	}
	return 1
}

func (b *Backend) SetCapacity(c int64) { b.capacity.Store(c) }

// NormalizedLoad is what least-outstanding compares (HIER-5).
//
// Comparing raw in-flight counts across heterogeneous backends is HIER-N1: a
// child router fronting 40 GPUs with 40 requests in flight would look "more
// loaded" than an idle leaf at 1, starving an entire healthy subtree. Dividing
// by capacity makes the comparison meaningful. For a fleet of uniform leaves
// with capacity 1 this degenerates to the raw count, at no cost.
func (b *Backend) NormalizedLoad() float64 {
	return float64(b.inflight.Load()) / float64(b.Capacity())
}

func (b *Backend) Health() Health     { return Health(b.health.Load()) }
func (b *Backend) SetHealth(h Health) { b.health.Store(int32(h)) }
func (b *Backend) Draining() bool     { return b.draining.Load() }

// Requests is the number of leases ever acquired on this backend.
func (b *Backend) Requests() uint64 { return b.requests.Load() }

// Down reports a backend that is unhealthy or being removed. Unknown is not
// down: a backend that has not been checked yet has not failed, and its
// counters are real. Used to decide when per-backend series stop being
// reported (see metrics.BackendSeries).
func (b *Backend) Down() bool { return b.Health() == Unhealthy || b.Draining() }

// Label is the backend's metric label: its canonical URL.
func (b *Backend) Label() string { return b.URL }

// Available reports whether this backend may receive new traffic.
//
// Note it reads CB.State(), which is read-only, and never CB.Allow(), which
// mutates. Filtering must not consume a half-open probe token for a backend it
// does not go on to select (R2).
func (b *Backend) Available() bool {
	return b.Health() == Healthy && !b.Draining() && b.CB.State() != circuit.Open
}

// AddInflight is the in-flight counter mutator.
//
// It is exported only because Go has no friend packages: the sole legitimate
// caller is internal/lease. Nothing else — not health checks, not admin
// endpoints, not discovery, not flush_cache — may call it (LB-6). This is
// enforced mechanically by hack/lease_fence_test.go rather than by convention,
// because "one increment, three decrements, plus a health checker that zeroes
// every counter every ten cycles" is the exact defect that motivated the
// rewrite (LB-N1, LB-N2, HLT-N5).
func (b *Backend) AddInflight(d int64) int64 { return b.inflight.Add(d) }

// CountRequest bumps the routed-request counter. Lease-only, for the same
// reason as AddInflight: one acquire, one count.
func (b *Backend) CountRequest() { b.requests.Add(1) }

// StoreInflight is likewise lease-only; it exists to clamp a detected
// underflow rather than let the counter wrap (LB-5).
func (b *Backend) StoreInflight(v int64) { b.inflight.Store(v) }

// Canonical normalizes a backend URL so that two inputs denoting the same
// backend compare equal (WRK-1): lowercase scheme and host, explicit port, no
// path, no trailing slash, IPv6 correctly bracketed (SD-9).
func Canonical(raw string) (string, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "", fmt.Errorf("empty backend URL")
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("parse %q: %w", raw, err)
	}
	scheme := strings.ToLower(u.Scheme)
	if scheme != "http" && scheme != "https" {
		// Rejecting file://, gopher:// and friends here is the SSRF guard for
		// the admin API (SEC-5).
		return "", fmt.Errorf("unsupported scheme %q (want http or https)", u.Scheme)
	}
	host := strings.ToLower(u.Hostname())
	if host == "" {
		return "", fmt.Errorf("missing host in %q", raw)
	}
	port := u.Port()
	if port == "" {
		if scheme == "https" {
			port = "443"
		} else {
			port = "80"
		}
	}
	// The PATH is part of the identity, not decoration. An endpoint can live
	// behind a base path — Google's OpenAI-compatible surface is
	// /v1beta/openai, and a vLLM behind an ingress prefix is the same shape —
	// and dropping it silently rewrites the backend to a different service on
	// the same host. Two paths on one host are two backends.
	path := strings.TrimRight(u.EscapedPath(), "/")
	return scheme + "://" + net.JoinHostPort(host, port) + path, nil
}

// ResolveURL builds the URL for a request the ROUTER makes to a backend on its
// own behalf — a health probe, a metrics scrape, a model listing — from the
// backend's configured URL and the path being asked for.
//
// It exists because Canonical preserves a backend's base path, and every caller
// that then wrote backendURL+path got it wrong in one of two directions. A base
// path is real for providers not hosted at the root (Google's OpenAI-compatible
// surface is /v1beta/openai), and a redundant one is common because people
// paste an OpenAI base_url ending in /v1. Plain concatenation turns the first
// into a correct URL only by accident and the second into /v1/v1/models.
//
// The rule is the same one the proxy applies to client requests: a base path
// REPLACES the version segment rather than stacking on it, because that is what
// base_url means everywhere it appears.
func ResolveURL(backendURL, path string) string {
	u, err := url.Parse(backendURL)
	if err != nil || u.Path == "" || u.Path == "/" {
		// Nothing to reconcile: the overwhelmingly common case, and the one
		// where concatenation was already right.
		return strings.TrimRight(backendURL, "/") + path
	}
	base := strings.TrimRight(u.Path, "/")
	u.Path = JoinBasePath(base, path)
	u.RawPath = ""
	return u.String()
}

// JoinBasePath composes a backend's base path with a request path. Exported so
// the proxy applies the identical rule to client traffic; see ResolveURL.
func JoinBasePath(base, path string) string {
	base = strings.TrimRight(base, "/")
	if base == "" {
		return path
	}
	if rest, ok := strings.CutPrefix(path, "/v1/"); ok {
		return base + "/" + rest
	}
	return base + path
}
