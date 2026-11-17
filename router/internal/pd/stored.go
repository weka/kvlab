package pd

import (
	"time"

	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/policy/affinity"
	"github.com/weka/kvlab/router/internal/registry"
)

// Stored records the prompts the shared KV tier holds, for dynamic P/D's
// charge: every prompt a prefill leg computed (or a decode node recomputed
// after a failed leg) was stored there, and any prefill node can load it.
//
// It is a prefix tree of its own with a single holder, the tier, so unlike
// the P and D flows' trees nothing in it is forgotten when a node changes
// role; only its tail TTL (--cache-tail-ttl) ages a prompt out.
type Stored struct {
	flow *affinity.Policy
	tier *registry.Backend
}

// NewStored builds the record on cfg's tree settings.
func NewStored(cfg affinity.Config) (*Stored, error) {
	cfg.CostRouting = false
	flow, err := affinity.New(cfg, policy.LeastOutstanding{})
	if err != nil {
		return nil, err
	}
	tier := &registry.Backend{URL: "tier://stored"}
	flow.AddBackend(tier)
	return &Stored{flow: flow, tier: tier}, nil
}

// Record notes rr's prompt as stored.
func (s *Stored) Record(rr *policy.RoutingRequest) {
	if s == nil {
		return
	}
	c := *rr
	c.PolicyState = nil
	s.flow.Commit(s.tier, &c)
}

// Depth is how many leading tokens of rr's prompt are recorded stored.
func (s *Stored) Depth(rr *policy.RoutingRequest) int64 {
	if s == nil {
		return 0
	}
	return s.flow.HeldDepth(rr)
}

// Sweep ages out prompts idle for the tail TTL; TailTTL paces it.
func (s *Stored) Sweep()                 { s.flow.Sweep() }
func (s *Stored) TailTTL() time.Duration { return s.flow.TailTTL() }

// PublishGauges publishes the record's size as the tree gauges of its pool.
func (s *Stored) PublishGauges() { s.flow.PublishGauges() }
