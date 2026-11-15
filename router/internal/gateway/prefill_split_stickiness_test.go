package gateway_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/clock"
	"github.com/weka/kvlab/router/internal/dialect/openai"
	"github.com/weka/kvlab/router/internal/gateway"
	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/mockvllm"
	"github.com/weka/kvlab/router/internal/policy/affinity"
	"github.com/weka/kvlab/router/internal/pool"
	"github.com/weka/kvlab/router/internal/prefill"
	"github.com/weka/kvlab/router/internal/proxy"
	"github.com/weka/kvlab/router/internal/registry"
	"github.com/weka/kvlab/router/internal/routing"
)

// These replay multi-turn conversations through the real gateway's prefill
// split under cost routing. Every turn adds more than the engines' rejection
// threshold, so every turn is refused by its decode node and prefilled as a job
// on a prefill node.

const (
	splitSysChars  = 64_000 // ~16k-token conversation prefix
	splitTurnChars = 12_000 // ~3k tokens per turn, over rejectAt
	splitTurnToks  = splitTurnChars / 4
)

// splitTurn is turn k of conversation s; each turn extends the previous one.
func splitTurn(s, k int) []byte {
	msgs := []map[string]any{{"role": "system", "content": fmt.Sprintf("conversation %04d ", s) + strings.Repeat(fmt.Sprintf("s%d ", s), splitSysChars/4)}}
	for j := 0; j <= k; j++ {
		msgs = append(msgs, map[string]any{"role": "user", "content": fmt.Sprintf("question %d ", j) + strings.Repeat("q ", splitTurnChars/2)})
		if j < k {
			msgs = append(msgs, map[string]any{"role": "assistant", "content": fmt.Sprintf("answer %d", j)})
		}
	}
	b, _ := json.Marshal(map[string]any{"model": "m", "max_tokens": 4, "messages": msgs})
	return b
}

type splitStickFleet struct {
	srv     *httptest.Server
	pool    *pool.Pool
	backend []*registry.Backend
	engines []*mockvllm.Engine
	prefill []int // node indexes in prefill mode

	mu    sync.Mutex
	calls []call
}

// newSplitStickFleet builds n engines, with a shared tier when tier is set,
// through pool.New under cost routing, and holds nPrefill of them in prefill
// mode.
func newSplitStickFleet(t *testing.T, n, nPrefill int, tier bool, storeUnit int64) *splitStickFleet {
	t.Helper()
	f := &splitStickFleet{}
	var shared *mockvllm.Tier
	if tier {
		shared = mockvllm.NewTier()
	}
	var cfgB []pool.Backend
	for i := range n {
		eng := mockvllm.NewEngine(mockvllm.Config{Tier: shared, PrefillRejectTokens: rejectAt, DefaultMaxTokens: 4})
		t.Cleanup(eng.Close)
		f.engines = append(f.engines, eng)
		h := mockvllm.NewServer(eng).Handler()
		node := i
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				body, _ := io.ReadAll(r.Body)
				r.Body = io.NopCloser(bytes.NewReader(body))
				var doc struct {
					MaxTokens float64        `json:"max_tokens"`
					XArgs     map[string]any `json:"vllm_xargs"`
				}
				_ = json.Unmarshal(body, &doc)
				f.mu.Lock()
				f.calls = append(f.calls, call{node: node, path: r.URL.Path, maxTokens: doc.MaxTokens,
					bypass: doc.XArgs["weka_prefill"] != nil})
				f.mu.Unlock()
			}
			h.ServeHTTP(w, r)
		}))
		t.Cleanup(ts.Close)
		cfgB = append(cfgB, pool.Backend{URL: ts.URL})
	}
	const budget = 1 << 30
	p, err := pool.New(pool.Config{
		Name: "default", Backends: cfgB, DefaultCapacity: 1, DefaultDialect: openai.New().ID(),
		Flow: affinity.Config{CostRouting: true},
		PrefillSplit: &prefill.Config{NodeBudgetTokens: budget, ReleaseIdle: time.Hour,
			StoreUnitTokens: storeUnit, MaxFleetFraction: 0.5},
	}, clock.Real{}, slog.Default())
	if err != nil {
		t.Fatal(err)
	}
	f.pool = p
	byURL := map[string]*registry.Backend{}
	for _, b := range p.Registry.Snapshot().Backends {
		b.SetHealth(registry.Healthy)
		byURL[b.URL] = b
	}
	for _, c := range cfgB {
		canon, _ := registry.Canonical(c.URL)
		f.backend = append(f.backend, byURL[canon])
	}

	// A job of the whole budget fills its node, so the next one converts
	// another: nPrefill of them leave nPrefill prefill nodes, which the long
	// idle release keeps.
	var jobs []*prefill.Job
	for range nPrefill {
		j, err := p.Split.Reserve(f.backend, budget, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.Split.WaitReady(context.Background(), j); err != nil {
			t.Fatal(err)
		}
		jobs = append(jobs, j)
	}
	for _, j := range jobs {
		j.Done()
	}
	for i, b := range f.backend {
		if p.Split.ModeOf(b) == prefill.Prefill {
			f.prefill = append(f.prefill, i)
		}
	}
	if len(f.prefill) != nPrefill {
		t.Fatalf("prefill nodes %v, want %d", f.prefill, nPrefill)
	}

	tbl, err := routing.NewTable([]routing.Rule{{Pool: p}})
	if err != nil {
		t.Fatal(err)
	}
	gw := gateway.New(gateway.Config{MaxBodyBytes: 64 << 20, DefaultCapacity: 1}, tbl,
		proxy.New(proxy.Config{MaxAttempts: 2}), openai.New())
	f.srv = httptest.NewServer(gw)
	t.Cleanup(f.srv.Close)
	return f
}

// send posts one turn and returns the node its prefill job ran on and what
// that node's engine counted for it: prompt, cached and tier tokens.
func (f *splitStickFleet) send(t *testing.T, s, k int) (node int, delta mockvllm.Stats) {
	t.Helper()
	before := make([]mockvllm.Stats, len(f.engines))
	for i, e := range f.engines {
		before[i] = e.Stats()
	}
	f.mu.Lock()
	first := len(f.calls)
	f.mu.Unlock()
	resp, err := http.Post(f.srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(splitTurn(s, k)))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("conversation %d turn %d: %d %s", s, k, resp.StatusCode, body)
	}
	node = -1
	f.mu.Lock()
	for _, c := range f.calls[first:] {
		if isPrefillJob(c) {
			node = c.node
		}
	}
	f.mu.Unlock()
	if node < 0 {
		t.Fatalf("conversation %d turn %d ran no prefill job", s, k)
	}
	after, b := f.engines[node].Stats(), before[node]
	return node, mockvllm.Stats{
		PromptTokens:   after.PromptTokens - b.PromptTokens,
		CachedTokens:   after.CachedTokens - b.CachedTokens,
		ExternalTokens: after.ExternalTokens - b.ExternalTokens,
	}
}

// splitSaturation keeps the prefill nodes busy with other conversations' jobs,
// as a saturated split is: before each request it redraws, per prefill node,
// base±jitter jobs of traffic outside the replay plus one job, with
// probability busy, for each conversation the node last served. A job is
// charged to the manager like a real one (a turn's uncached tokens) and holds a
// lease like a dispatched one (its context as active and prefill load).
type splitSaturation struct {
	f            *splitStickFleet
	rnd          *rand.Rand
	busy         float64
	base, jitter int
	ctxTokens    int64
	jobs         []*prefill.Job
	leases       []*lease.Lease
}

func (s *splitSaturation) release() {
	for _, j := range s.jobs {
		j.Done()
	}
	for _, l := range s.leases {
		l.Release()
	}
	s.jobs, s.leases = s.jobs[:0], s.leases[:0]
}

func (s *splitSaturation) redraw(t *testing.T, last []int, skip int) {
	t.Helper()
	s.release()
	legs := map[int]int{}
	for _, i := range s.f.prefill {
		legs[i] = s.base
		if s.jitter > 0 {
			legs[i] += s.rnd.Intn(2*s.jitter+1) - s.jitter
		}
	}
	for c, n := range last {
		if c != skip && n >= 0 && s.rnd.Float64() < s.busy {
			legs[n]++
		}
	}
	for i, k := range legs {
		b := s.f.backend[i]
		for range k {
			j, err := s.f.pool.Split.Reserve([]*registry.Backend{b}, splitTurnToks, nil)
			if err != nil || j.Backend != b {
				t.Fatalf("background job on node %d: %v", i, err)
			}
			l := lease.Acquire(b)
			l.StartActive(s.ctxTokens)
			l.StartPrefill(s.ctxTokens)
			s.jobs, s.leases = append(s.jobs, j), append(s.leases, l)
		}
	}
}

type splitReplay struct {
	stay, moves              int
	prompt, cached, external int64
	perNode                  map[int]int
}

func (r splitReplay) stayRate() float64 { return float64(r.stay) / float64(r.moves) }

// replaySplit runs nConv conversations of nTurns turns, interleaved turn by
// turn, with the prefill nodes saturated, and reports how often a job went to
// the node that ran the conversation's previous job, and what the job nodes'
// engines found cached for turns after the first.
func replaySplit(t *testing.T, f *splitStickFleet, nConv, nTurns int) splitReplay {
	t.Helper()
	sat := &splitSaturation{f: f, rnd: rand.New(rand.NewSource(1)),
		busy: 0.5, base: 3, jitter: 2, ctxTokens: 20_000}
	defer sat.release()
	last := make([]int, nConv)
	for i := range last {
		last[i] = -1
	}
	r := splitReplay{perNode: map[int]int{}}
	for k := range nTurns {
		for s := range nConv {
			sat.redraw(t, last, s)
			node, d := f.send(t, s, k)
			r.perNode[node]++
			if k > 0 {
				r.moves++
				if node == last[s] {
					r.stay++
				}
				r.prompt += d.PromptTokens
				r.cached += d.CachedTokens
				r.external += d.ExternalTokens
			}
			last[s] = node
		}
	}
	t.Logf("prefill-node stays %d/%d (%.2f); turns 1+ on the job node: cached %.3f of prompt (local %.3f, tier %.3f); jobs per prefill node %v",
		r.stay, r.moves, r.stayRate(), float64(r.cached)/float64(r.prompt),
		float64(r.cached-r.external)/float64(r.prompt), float64(r.external)/float64(r.prompt), r.perNode)
	for i, c := range r.perNode {
		if share := nConv * nTurns / len(f.prefill); c > 2*share {
			t.Errorf("prefill node %d ran %d jobs, more than twice its share %d (%v)", i, c, share, r.perNode)
		}
	}
	return r
}

// Under cost routing with every prefill node saturated, and nothing in a tier
// to load a conversation's earlier turns from, a conversation's next job must
// go to the prefill node holding its prefix: any other recomputes all of it.
func TestSplitCostRoutingKeepsConversationsOnTheirPrefillNodeUnderSaturation(t *testing.T) {
	// No tier, and a storage cut no prompt reaches: nothing is loadable, so the
	// only copy of a prefix is on the node that computed it.
	f := newSplitStickFleet(t, 12, 4, false, 1<<40)
	if r := replaySplit(t, f, 16, 4); r.stayRate() < 0.95 {
		t.Errorf("prefill-node stickiness %.2f: jobs leave the node holding their prefix for one that recomputes it", r.stayRate())
	}
}

// With the tier holding every earlier job's prompt, any prefill node loads a
// conversation's prefix, so jobs follow the shortest queue: they still spread
// evenly, and every job node finds the earlier turns cached, locally or in the
// tier.
func TestSplitCostRoutingSpreadsByQueueWhenTheTierHoldsThePrefix(t *testing.T) {
	f := newSplitStickFleet(t, 12, 4, true, 0)
	r := replaySplit(t, f, 16, 4)
	if hit := float64(r.cached) / float64(r.prompt); hit < 0.75 {
		t.Errorf("job nodes found %.3f of turns 1+ cached; the earlier turns are all in the tier", hit)
	}
}

// A prefix an earlier job stored in the tier is loadable by every prefill node,
// so the node holding it locally keeps no advantage for it: with a queue
// shorter than the prefix it loses to an idle node. With nothing stored the
// same queue is worth waiting behind, and the job stays.
func TestSplitJobIsCreditedForThePrefixAnEarlierJobStored(t *testing.T) {
	for _, c := range []struct {
		name      string
		storeUnit int64
		moves     bool
	}{
		{"whole prompt stored", 0, true},
		{"nothing stored", 1 << 40, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newSplitStickFleet(t, 4, 2, true, c.storeUnit)
			holder, _ := f.send(t, 0, 0)
			// A queue of half the ~19k-token prefix on the holder.
			j, err := f.pool.Split.Reserve([]*registry.Backend{f.backend[holder]}, 8_000, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer j.Done()
			got, _ := f.send(t, 0, 1)
			if moved := got != holder; moved != c.moves {
				t.Errorf("turn 1 job on node %d after turn 0 on %d: moved=%v, want %v", got, holder, moved, c.moves)
			}
		})
	}
}
