package affinity

import (
	"github.com/weka/wekai/router/internal/metrics"
	"github.com/weka/wekai/router/internal/policy"
	"github.com/weka/wekai/router/internal/registry"
)

// causeSet records which band signals got an affinity holder skipped in one
// decision, so a split can be attributed (router_split_trigger_total) and the
// relative signals' eligibility applied to its target.
type causeSet uint8

const (
	causeConcurrency causeSet = 1 << iota
	causeSoftTTFT
	causeRelTTFT
	causeRelDecode
)

// count increments router_split_trigger_total once per cause present.
func (c causeSet) count() {
	for bit, label := range map[causeSet]string{
		causeConcurrency: "concurrency", causeSoftTTFT: "soft_ttft",
		causeRelTTFT: "relative_ttft", causeRelDecode: "relative_decode",
	} {
		if c&bit != 0 {
			metrics.SplitTriggers.WithLabelValues(label).Inc()
		}
	}
}

// lowestOther is the lowest f over usable backends other than self; ok is
// false when there is no other backend to compare against.
func lowestOther(usable []*registry.Backend, self *registry.Backend, f func(*registry.Backend) int64) (int64, bool) {
	var lo int64
	ok := false
	for _, o := range usable {
		if o == self {
			continue
		}
		if v := f(o); !ok || v < lo {
			lo, ok = v, true
		}
	}
	return lo, ok
}

// estTokens is this request's estimated context size in tokens: the gateway's
// body-based estimate when present, else the sum of its routing units.
func estTokens(rr *policy.RoutingRequest) int64 {
	if rr.EstTokens > 0 {
		return rr.EstTokens
	}
	var n int64
	for _, u := range rr.Units {
		n += int64(u.Tokens)
	}
	return n
}
