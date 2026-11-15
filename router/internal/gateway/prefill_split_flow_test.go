package gateway_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
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

// A split fleet built the way the router builds it (pool.New), under cost
// routing, so the pool carries its separate prefill flow.
type costSplitFleet struct {
	srv     *httptest.Server
	pool    *pool.Pool
	backend []*registry.Backend

	mu    sync.Mutex
	calls []call
}

func newCostSplitFleet(t *testing.T, n int) *costSplitFleet {
	t.Helper()
	f := &costSplitFleet{}
	tier := mockvllm.NewTier()
	var cfgB []pool.Backend
	for i := range n {
		eng := mockvllm.NewEngine(mockvllm.Config{Tier: tier, PrefillRejectTokens: rejectAt, DefaultMaxTokens: 4})
		t.Cleanup(eng.Close)
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
	p, err := pool.New(pool.Config{
		Name: "default", Backends: cfgB, DefaultCapacity: 1, DefaultDialect: openai.New().ID(),
		Flow: affinity.Config{CostRouting: true},
		// A tiny budget makes a second concurrent job convert a second node;
		// a long idle release keeps converted nodes in prefill mode.
		PrefillSplit: &prefill.Config{NodeBudgetTokens: 1, ReleaseIdle: time.Hour},
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

// twoPrefillNodes converts and holds two nodes in prefill mode.
func (f *costSplitFleet) twoPrefillNodes(t *testing.T) {
	t.Helper()
	ctx := context.Background()
	j1, err := f.pool.Split.Reserve(f.backend, 1<<20, nil)
	if err != nil {
		t.Fatal(err)
	}
	j2, err := f.pool.Split.Reserve(f.backend, 1<<20, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, j := range []*prefill.Job{j1, j2} {
		if err := f.pool.Split.WaitReady(ctx, j); err != nil {
			t.Fatal(err)
		}
	}
	j1.Done()
	j2.Done()
	if j1.Backend == j2.Backend {
		t.Fatal("expected two distinct prefill nodes")
	}
}

func (f *costSplitFleet) turn(t *testing.T, convo string, turn int) (jobNode, decodeNode int) {
	t.Helper()
	msgs := []map[string]any{}
	for k := 0; k <= turn; k++ {
		msgs = append(msgs, map[string]any{"role": "user", "content": heavyPrompt(convo + string(rune('a'+k)))})
		if k < turn {
			msgs = append(msgs, map[string]any{"role": "assistant", "content": "ok"})
		}
	}
	b, _ := json.Marshal(map[string]any{"model": "m", "max_tokens": 4, "messages": msgs})
	f.mu.Lock()
	before := len(f.calls)
	f.mu.Unlock()
	resp, err := http.Post(f.srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(b)))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("%s turn %d: %d %s", convo, turn, resp.StatusCode, body)
	}
	jobNode, decodeNode = -1, -1
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, c := range f.calls[before:] {
		switch {
		case isPrefillJob(c):
			jobNode = c.node
		case isDecode(c):
			decodeNode = c.node
		}
	}
	return jobNode, decodeNode
}

// With several prefill nodes, a conversation's jobs keep going to the prefill
// node that holds its prefix, chosen by the prefill flow's own tree; and its
// decode resends keep going to one decode node, chosen by the decode tree.
func TestSplitPrefillAndDecodeStickinessPerConversationUnderCostRouting(t *testing.T) {
	f := newCostSplitFleet(t, 4)
	f.twoPrefillNodes(t)

	jobs := map[string]map[int]bool{"X": {}, "Y": {}}
	decodes := map[string]map[int]bool{"X": {}, "Y": {}}
	record := func(c string, turn int) {
		j, d := f.turn(t, c, turn)
		if j < 0 || d < 0 {
			t.Fatalf("%s turn %d: job node %d, decode node %d; want both legs", c, turn, j, d)
		}
		if f.pool.Split.ModeOf(f.backend[d]) != prefill.Decode {
			t.Fatalf("decode resend landed on a non-decode node %d", d)
		}
		jobs[c][j], decodes[c][d] = true, true
	}
	record("X", 0)
	// Y's first job must not tie with X's: make X's prefill node busier.
	xNode := -1
	for n := range jobs["X"] {
		xNode = n
	}
	held := lease.Acquire(f.backend[xNode])
	record("Y", 0)
	held.Release()
	if jobs["Y"][xNode] {
		t.Fatalf("Y's first job went to X's busier node %d", xNode)
	}
	for turn := 1; turn < 4; turn++ {
		record("X", turn)
		record("Y", turn)
	}
	for c := range jobs {
		if len(jobs[c]) != 1 {
			t.Errorf("conversation %s moved between prefill nodes: %v", c, jobs[c])
		}
		if len(decodes[c]) != 1 {
			t.Errorf("conversation %s moved between decode nodes: %v", c, decodes[c])
		}
	}
}
