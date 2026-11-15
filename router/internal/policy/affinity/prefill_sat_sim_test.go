package affinity

import (
	"container/heap"
	"context"
	"fmt"
	"math/rand/v2"
	"sort"
	"testing"
	"time"

	"github.com/weka/kvlab/kvcache"
	"github.com/weka/kvlab/router/internal/clock"
	"github.com/weka/kvlab/router/internal/lease"
	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/registry"
)

// This is a discrete-event replay of the prefill leg of P/D disaggregation
// against the real cost policy, sized like a live agentic fleet: prompts of
// 50k-250k tokens growing by 1-3k per turn, a dozen turns per conversation, and
// P nodes that are compute-bound with no shared KV tier. Each P node drains its
// FIFO queue at a fixed rate of uncached tokens, so a recompute a decision
// causes really delays every leg queued behind it, and the conversations are a
// closed loop, so a slower fleet sends its next turns later. That feedback is
// what a replay of random queue lengths cannot show: moves that inject
// recomputes lengthen the queues that make the next moves look worthwhile.
//
// The policy sees what the router sees: leases charging its own estimates
// (active context, uncached prefill) from dispatch to completion, and the commit
// when the leg completes, which is when a P node's response arrives.

type prefillSimConfig struct {
	seed                 uint64
	nodes                int
	rate                 float64 // uncached tokens/s per P node
	conversations        int     // live at once; a finished one is replaced
	turns                int
	minPrompt, maxPrompt int // first-turn prompt, tokens
	minDelta, maxDelta   int // growth per turn, tokens
	decode               time.Duration
	duration, warmup     time.Duration
	unitTokens           int
	// arrivals, when set, starts conversations as a Poisson process at this
	// rate per second instead of keeping `conversations` live: the open loop
	// of a replayed trace, which keeps offering work however slow the fleet is.
	arrivals float64
	flow     Config
}

func livePrefillSim() prefillSimConfig {
	return prefillSimConfig{
		seed: 7, nodes: 8, rate: 28_000, conversations: 28 * 8, turns: 12,
		minPrompt: 50_000, maxPrompt: 250_000, minDelta: 1_000, maxDelta: 3_000,
		decode: 3 * time.Second, duration: 40 * time.Minute, warmup: 10 * time.Minute,
		unitTokens: 500,
		flow:       Config{CostRouting: true},
	}
}

type simConv struct {
	id     int
	prompt int // current turn's prompt, tokens
	turn   int
	last   int // P node of the previous turn, -1 before the first
}

type simEvent struct {
	at   time.Duration
	seq  int
	conv *simConv
	// leg is set for a completion, nil for an issue.
	leg *simLeg
}

type simLeg struct {
	node     int
	rr       *policy.RoutingRequest
	lease    *lease.Lease
	uncached int
}

type simEvents []simEvent

func (e simEvents) Len() int { return len(e) }
func (e simEvents) Less(i, j int) bool {
	if e[i].at != e[j].at {
		return e[i].at < e[j].at
	}
	return e[i].seq < e[j].seq
}
func (e simEvents) Swap(i, j int) { e[i], e[j] = e[j], e[i] }
func (e *simEvents) Push(x any)   { *e = append(*e, x.(simEvent)) }
func (e *simEvents) Pop() any {
	old := *e
	x := old[len(old)-1]
	*e = old[:len(old)-1]
	return x
}

type prefillSimResult struct {
	legs, stays, moves, newConvs int
	prompt, cached               int64
	perNodeNew                   []int
	ttft                         []time.Duration
	peakQueue                    float64 // largest node backlog seen, seconds
}

func (r prefillSimResult) stickiness() float64 {
	return float64(r.stays) / float64(r.stays+r.moves)
}
func (r prefillSimResult) hit() float64 { return float64(r.cached) / float64(r.prompt) }
func (r prefillSimResult) ttftP50() time.Duration {
	if len(r.ttft) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), r.ttft...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return s[len(s)/2]
}

func (r prefillSimResult) String() string {
	return fmt.Sprintf("legs %d, stickiness %.3f (%d stay / %d move), P cache hit %.3f, ttft p50 %v, peak backlog %.0fs, new conversations per node %v",
		r.legs, r.stickiness(), r.stays, r.moves, r.hit(), r.ttftP50().Round(100*time.Millisecond), r.peakQueue, r.perNodeNew)
}

func runPrefillSim(t *testing.T, cfg prefillSimConfig) prefillSimResult {
	t.Helper()
	clk := clock.NewFake(time.Time{})
	start := clk.Now()
	fc := cfg.flow
	fc.Clock = clk
	if fc.TailTTL == 0 {
		fc.TailTTL = time.Hour
	}
	p, err := New(fc, nil)
	if err != nil {
		t.Fatal(err)
	}
	reg := registry.New(registry.Options{})
	var nodes []*registry.Backend
	for i := range cfg.nodes {
		b, err := reg.Add(registry.Spec{URL: fmt.Sprintf("http://p%02d:8000", i), Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		b.SetHealth(registry.Healthy)
		p.AddBackend(b)
		nodes = append(nodes, b)
	}
	rng := rand.New(rand.NewPCG(cfg.seed, cfg.seed+1))
	busyUntil := make([]time.Duration, cfg.nodes)
	// held[node][conv] is the prompt length of the conversation the node last
	// computed: its ground-truth cache, with no eviction.
	held := make([]map[int]int, cfg.nodes)
	for i := range held {
		held[i] = map[int]int{}
	}

	var ev simEvents
	seq := 0
	push := func(at time.Duration, c *simConv, leg *simLeg) {
		seq++
		heap.Push(&ev, simEvent{at: at, seq: seq, conv: c, leg: leg})
	}
	nextID := 0
	newConv := func(at time.Duration) {
		nextID++
		c := &simConv{id: nextID, last: -1,
			prompt: cfg.minPrompt + rng.IntN(cfg.maxPrompt-cfg.minPrompt+1)}
		push(at, c, nil)
	}
	if cfg.arrivals > 0 {
		for at := time.Duration(0); at < cfg.duration; {
			newConv(at)
			at += time.Duration(rng.ExpFloat64() / cfg.arrivals * float64(time.Second))
		}
	} else {
		for i := range cfg.conversations {
			// Stagger the start over the first minute, as a client ramps up.
			newConv(time.Duration(i) * time.Minute / time.Duration(cfg.conversations))
		}
	}
	unitsOf := func(c *simConv) []kvcache.Unit {
		n := (c.prompt + cfg.unitTokens - 1) / cfg.unitTokens
		u := make([]kvcache.Unit, n)
		for j := range u {
			u[j] = kvcache.Unit{Hash: uint64(c.id)<<32 | uint64(j+1), Tokens: int32(cfg.unitTokens)}
		}
		return u
	}

	res := prefillSimResult{perNodeNew: make([]int, cfg.nodes)}
	for ev.Len() > 0 {
		e := heap.Pop(&ev).(simEvent)
		if e.at > cfg.duration {
			break
		}
		clk.Advance(start.Add(e.at).Sub(clk.Now()))
		now := e.at
		c := e.conv
		if e.leg != nil {
			l := e.leg
			l.lease.Release()
			p.Commit(nodes[l.node], l.rr)
			held[l.node][c.id] = c.prompt
			c.last = l.node
			c.turn++
			if c.turn >= cfg.turns {
				if cfg.arrivals <= 0 {
					newConv(now + cfg.decode)
				}
				continue
			}
			c.prompt += cfg.minDelta + rng.IntN(cfg.maxDelta-cfg.minDelta+1)
			push(now+cfg.decode, c, nil)
			continue
		}

		rr := &policy.RoutingRequest{Units: unitsOf(c), Model: "m", PrefillOnly: true}
		b, err := p.Select(context.Background(), nodes, rr)
		if err != nil {
			t.Fatalf("select: %v", err)
		}
		n := 0
		for i, x := range nodes {
			if x == b {
				n = i
			}
		}
		total := len(rr.Units) * cfg.unitTokens
		uncached := total - min(held[n][c.id], total)
		lse := lease.Acquire(b)
		lse.StartActive(int64(total))
		lse.StartPrefill(rr.Prefill.Tokens)
		begin := max(now, busyUntil[n])
		done := begin + time.Duration(float64(uncached)/cfg.rate*float64(time.Second))
		busyUntil[n] = done
		res.peakQueue = max(res.peakQueue, (done - now).Seconds())
		push(done, c, &simLeg{node: n, rr: rr, lease: lse, uncached: uncached})

		if now < cfg.warmup {
			continue
		}
		res.legs++
		res.prompt += int64(total)
		res.cached += int64(total - uncached)
		res.ttft = append(res.ttft, done-now)
		switch {
		case c.last < 0:
			res.newConvs++
			res.perNodeNew[n]++
		case n == c.last:
			res.stays++
		default:
			res.moves++
		}
	}
	return res
}

// openLoop sets conversations to arrive at load times the rate the fleet could
// serve if every turn stayed on its holder.
func (c prefillSimConfig) openLoop(load float64) prefillSimConfig {
	work := float64(c.minPrompt+c.maxPrompt)/2 + float64(c.turns)*float64(c.minDelta+c.maxDelta)/2
	c.arrivals = load * c.rate * float64(c.nodes) / work
	return c
}

func (c prefillSimConfig) weighted(w float64) prefillSimConfig {
	c.flow.MoveWeight = w
	return c
}

// A greedy time-to-first-token choice (move weight 1) is unstable on this
// fleet: offered 80% of what it could serve with every turn on its holder, it
// moves a third of the turns, and the recomputes those moves inject overload it,
// with minutes of queue. The default weight keeps turns on their holders, and
// the same offered load is served with seconds of queue at the cache hit rate
// the workload allows. New conversations, which hold nothing anywhere, still
// spread by queue.
func TestPrefillLegStaysOnItsHolderOnACompoundingFleet(t *testing.T) {
	base := livePrefillSim()
	greedy := runPrefillSim(t, base.openLoop(0.8).weighted(1))
	t.Logf("open loop 0.8, weight 1:       %v", greedy)
	for _, c := range []struct {
		name string
		cfg  prefillSimConfig
	}{
		{"open loop 0.8", base.openLoop(0.8)},
		{"open loop 0.95", base.openLoop(0.95)},
		{"closed loop", base},
	} {
		r := runPrefillSim(t, c.cfg)
		t.Logf("%-15s default weight: %v", c.name+",", r)
		if r.stickiness() < 0.9 {
			t.Errorf("%s: stickiness %.3f, want >= 0.9", c.name, r.stickiness())
		}
		if r.hit() < 0.8 {
			t.Errorf("%s: P cache hit %.3f, want >= 0.8", c.name, r.hit())
		}
		share := float64(r.newConvs) / float64(base.nodes)
		for i, n := range r.perNodeNew {
			if float64(n) > 2*share {
				t.Errorf("%s: node %d took %d new conversations, over twice its share %.0f", c.name, i, n, share)
			}
		}
	}
	r := runPrefillSim(t, base.openLoop(0.8))
	if greedy.stickiness() > 0.8 || greedy.hit() > r.hit()-0.1 || greedy.legs > r.legs*2/3 {
		t.Errorf("weight 1 did not spread: %v; the replay no longer reproduces the instability it guards against", greedy)
	}
}

// A holder is left once its queue exceeds another node's by more than the move
// weight times the recompute the move causes, and not before.
func TestPrefillLegShedsAHolderOnlyPastTheMoveWeight(t *testing.T) {
	u := units(1, 2, 3, 4) // 1024 tokens; turn 2 adds 256
	next := append(units(1, 2, 3, 4), units(5)...)
	moveCost := int64(DefaultMoveWeight * 1024)
	for _, c := range []struct {
		name  string
		queue int64
		moves bool
	}{
		{"queue below the weighted move cost", moveCost - 1, false},
		{"queue above the weighted move cost", moveCost + 1, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := newCostPolicy(t)
			f := fleet(t, 2)
			for _, b := range f {
				p.AddBackend(b)
			}
			p.Commit(f[0], &policy.RoutingRequest{Units: u, Model: "m", PrefillOnly: true})
			f[0].AddPrefillTokens(c.queue)
			got, err := p.Select(context.Background(), f, &policy.RoutingRequest{Units: next, Model: "m", PrefillOnly: true})
			if err != nil {
				t.Fatal(err)
			}
			if moved := got != f[0]; moved != c.moves {
				t.Errorf("holder queue %d: moved=%v, want %v", c.queue, moved, c.moves)
			}
		})
	}
}
