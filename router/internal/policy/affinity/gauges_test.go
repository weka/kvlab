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
