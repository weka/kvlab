package affinity

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/registry"
)

// Under P/D the decode and prefill flows each hold a tree over the same
// backends. Their per-backend gauges are separate series, so one flow's
// publish cannot overwrite the other's.
func TestPerBackendCacheGaugesAreLabelledByFlow(t *testing.T) {
	pool := t.Name()
	decode, err := New(Config{PoolName: pool}, policy.LeastOutstanding{})
	if err != nil {
		t.Fatal(err)
	}
	prefill, err := New(Config{PoolName: pool + ":prefill"}, policy.LeastOutstanding{})
	if err != nil {
		t.Fatal(err)
	}
	reg := registry.New(registry.Options{})
	b, _ := reg.Add(registry.Spec{URL: "http://gauge-node", Prov: registry.ProvStatic, Capacity: 1})
	decode.AddBackend(b)
	prefill.AddBackend(b)
	prefill.Commit(b, &policy.RoutingRequest{Units: units(1, 2, 3)})

	prefill.PublishGauges()
	decode.PublishGauges() // the decode tree holds nothing on this node

	if got := testutil.ToFloat64(metrics.CacheEntries.WithLabelValues(pool+":prefill", b.URL)); got != 3 {
		t.Errorf("prefill flow entries = %v, want 3", got)
	}
	if got := testutil.ToFloat64(metrics.CacheEntries.WithLabelValues(pool, b.URL)); got != 0 {
		t.Errorf("decode flow entries = %v, want 0", got)
	}
}

func TestHeldDepthsAndBackendTokensArePerBackend(t *testing.T) {
	p, err := New(Config{PoolName: t.Name()}, policy.LeastOutstanding{})
	if err != nil {
		t.Fatal(err)
	}
	reg := registry.New(registry.Options{})
	a, _ := reg.Add(registry.Spec{URL: "http://depth-a", Prov: registry.ProvStatic, Capacity: 1})
	b, _ := reg.Add(registry.Spec{URL: "http://depth-b", Prov: registry.ProvStatic, Capacity: 1})
	c, _ := reg.Add(registry.Spec{URL: "http://depth-c", Prov: registry.ProvStatic, Capacity: 1})
	for _, x := range []*registry.Backend{a, b, c} {
		p.AddBackend(x)
	}
	p.Commit(a, req(units(1, 2, 3)))
	p.Commit(b, req(units(1)))
	got := p.HeldDepths(req(units(1, 2, 3, 4)), []*registry.Backend{a, b, c})
	if got[0] != 768 || got[1] != 256 || got[2] != 0 {
		t.Errorf("depths = %v, want [768 256 0]", got)
	}
	toks := p.BackendTokens()
	if toks[a.URL] != 768 || toks[b.URL] != 256 || toks[c.URL] != 0 {
		t.Errorf("backend tokens = %v", toks)
	}
}
