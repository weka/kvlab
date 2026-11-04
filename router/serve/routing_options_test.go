package serve

import (
	"strings"
	"testing"
	"time"
)

func TestValidateRouting(t *testing.T) {
	ladderOnly := []struct {
		flag string
		set  func(*Options)
	}{
		{"--soft-node-concurrency", func(o *Options) { o.SoftNodeConcurrency = 4 }},
		{"--soft-ttft-split", func(o *Options) { o.TTFTSoft = time.Second }},
		{"--relative-ttft-split", func(o *Options) { o.RelativeTTFT = true }},
		{"--relative-decode-split", func(o *Options) { o.RelativeDecode = true }},
		{"--transient-fallback-threshold", func(o *Options) { o.TransientFallback = 0.05 }},
		{"--cache-split-guard", func(o *Options) { o.SplitGuard = 0.3 }},
		{"--prefill-split", func(o *Options) { o.PrefillSplit = true }},
		{"--pd-prefill-backends", func(o *Options) { o.PDPrefillBackends = []string{"http://p:8000"} }},
	}
	for _, c := range ladderOnly {
		var cost, ladder Options
		cost.CostRouting = true
		c.set(&cost)
		c.set(&ladder)
		err := cost.validateRouting()
		if err == nil || !strings.Contains(err.Error(), c.flag) || !strings.Contains(err.Error(), "--no-cost-routing") {
			t.Errorf("%s under cost routing: err = %v, want it to name the flag and --no-cost-routing", c.flag, err)
		}
		if err := ladder.validateRouting(); err != nil {
			t.Errorf("%s under the ladder: %v", c.flag, err)
		}
	}

	allowed := []struct {
		flag string
		set  func(*Options)
	}{
		{"--max-node-concurrency", func(o *Options) { o.NodeConcurrency = 8 }},
		{"--max-node-concurrency-queue-depth", func(o *Options) { o.MaxNodeQueueDepth = 16 }},
		{"--hard-ttft-split", func(o *Options) { o.TTFTHard = time.Second }},
		{"--rebalance-ratio", func(o *Options) { o.RebalanceRatio = 0.5 }},
		{"--retry-time-limit", func(o *Options) { o.RetryTimeLimit = time.Second }},
		{"--cache-refusal-ttl", func(o *Options) { o.RefusalTTL = time.Second }},
		{"--cache-tail-ttl", func(o *Options) { o.TailTTL = time.Minute }},
		{"--cache-split-guard at its default", func(o *Options) { o.SplitGuard = DefaultSplitGuard }},
	}
	for _, c := range allowed {
		o := Options{CostRouting: true}
		c.set(&o)
		if err := o.validateRouting(); err != nil {
			t.Errorf("%s under cost routing: %v", c.flag, err)
		}
	}
}
