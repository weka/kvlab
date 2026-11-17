package serve

import (
	"strings"
	"testing"
	"time"
)

func TestValidatePD(t *testing.T) {
	routes := []Route{{Patterns: "*", Endpoints: []string{"http://p:8000", "http://d1:8000", "http://d2:8000"}}}
	cases := []struct {
		name string
		o    Options
		want string // substring of the error; empty is valid
	}{
		{"off", Options{Routes: routes}, ""},
		{"subset", Options{Routes: routes, PDPrefillBackends: []string{"http://p:8000/"}}, ""},
		{"with prefill split", Options{Routes: routes, PrefillSplit: true, PDPrefillBackends: []string{"http://p:8000"}}, "mutually exclusive"},
		{"not a backend", Options{Routes: routes, PDPrefillBackends: []string{"http://x:8000"}}, "not one of"},
		{"no decode left", Options{Routes: routes, PDPrefillBackends: []string{"http://p:8000", "http://d1:8000", "http://d2:8000"}}, "no decode backend"},
	}
	for _, c := range cases {
		err := c.o.validatePD()
		switch {
		case c.want == "" && err != nil:
			t.Errorf("%s: %v", c.name, err)
		case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
			t.Errorf("%s: err = %v, want %q", c.name, err, c.want)
		}
	}
}

// A pods: route has no static endpoint list, so the named prefill URLs cannot be
// checked against it at startup and every one goes to the pool, which matches
// them as pods appear. A purely static list keeps the hard error.
func TestValidatePDWithDiscovery(t *testing.T) {
	pdURL := []string{"http://10.10.41.83:8000"}
	disc := []Route{{Patterns: "*", DiscoverySelector: "app=vllm"}}
	if err := (Options{Routes: disc, PDPrefillBackends: pdURL}).validatePD(); err != nil {
		t.Errorf("discovery route: %v", err)
	}
	o := Options{Routes: disc, PDPrefillBackends: pdURL}
	if got := o.pdPrefillFor(disc[0]); len(got) != 1 || got[0] != pdURL[0] {
		t.Errorf("pdPrefillFor(discovery route) = %v, want %v", got, pdURL)
	}
	static := []Route{{Patterns: "*", Endpoints: []string{"http://d1:8000"}}}
	err := (Options{Routes: static, PDPrefillBackends: pdURL}).validatePD()
	if err == nil || !strings.Contains(err.Error(), "not one of") {
		t.Errorf("static list: err = %v, want the not-one-of error", err)
	}
}

func TestPDBandOverrides(t *testing.T) {
	routes := []Route{{Patterns: "*", Endpoints: []string{"http://p:8000", "http://d1:8000"}}}
	pd := []string{"http://p:8000"}
	cases := []struct {
		name string
		o    Options
		want string
	}{
		{"decode band wider than global", Options{Routes: routes, PDPrefillBackends: pd, NodeConcurrency: 16, SoftNodeConcurrency: 8, PDDecodeMaxNodeConcurrency: 64, PDDecodeSoftNodeConcurrency: 32}, ""},
		{"soft inherited from global", Options{Routes: routes, PDPrefillBackends: pd, NodeConcurrency: 16, SoftNodeConcurrency: 8, PDPrefillMaxNodeConcurrency: 12}, ""},
		{"prefill soft at max", Options{Routes: routes, PDPrefillBackends: pd, PDPrefillSoftNodeConcurrency: 8, PDPrefillMaxNodeConcurrency: 8}, "prefill soft"},
		{"decode max below inherited soft", Options{Routes: routes, PDPrefillBackends: pd, NodeConcurrency: 16, SoftNodeConcurrency: 8, PDDecodeMaxNodeConcurrency: 6}, "decode soft"},
		{"soft without any max", Options{Routes: routes, PDPrefillBackends: pd, PDDecodeSoftNodeConcurrency: 4}, "needs a max"},
		{"negative", Options{Routes: routes, PDPrefillBackends: pd, PDDecodeMaxNodeConcurrency: -1}, "negative"},
		{"without pd", Options{Routes: routes, PDDecodeMaxNodeConcurrency: 32}, "needs --pd-prefill-backends"},
	}
	for _, c := range cases {
		err := c.o.validatePD()
		switch {
		case c.want == "" && err != nil:
			t.Errorf("%s: %v", c.name, err)
		case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
			t.Errorf("%s: err = %v, want %q", c.name, err, c.want)
		}
	}
}

func TestPDRoleFlowsInheritAndOverride(t *testing.T) {
	o := Options{PDPrefillBackends: []string{"http://p:8000"}, NodeConcurrency: 16, SoftNodeConcurrency: 8,
		PDDecodeMaxNodeConcurrency: 64}
	dec, pre := o.roleFlowConfigs()
	if pre.NodeConcurrency != 16 || pre.SoftNodeConcurrency != 8 {
		t.Errorf("prefill band = %d..%d, want inherited 8..16", pre.SoftNodeConcurrency, pre.NodeConcurrency)
	}
	if dec.NodeConcurrency != 64 || dec.SoftNodeConcurrency != 8 {
		t.Errorf("decode band = %d..%d, want 8..64", dec.SoftNodeConcurrency, dec.NodeConcurrency)
	}
	if got, want := o.signalsLog(), "prefill=[refused,concurrency=8..16] decode=[refused,concurrency=8..64]"; got != want {
		t.Errorf("signalsLog = %q, want %q", got, want)
	}

	o.PDPrefillBackends = nil
	dec, pre = o.roleFlowConfigs()
	if dec.NodeConcurrency != 16 || pre.NodeConcurrency != 16 {
		t.Error("overrides applied without --pd-prefill-backends")
	}
	if got, want := o.signalsLog(), "refused,concurrency=8..16"; got != want {
		t.Errorf("signalsLog = %q, want %q", got, want)
	}
}

func TestRelativeThreadedIntoFlowConfig(t *testing.T) {
	c := Options{RelativeTTFT: true, RelativeDecode: true}.flowConfig()
	if !c.RelativeTTFT || !c.RelativeDecode {
		t.Errorf("flowConfig dropped the ratios: %+v", c)
	}
	if s := signalSummary(c); !strings.Contains(s, "relative-ttft") || !strings.Contains(s, "relative-decode") {
		t.Errorf("startup summary missing relative signals: %s", s)
	}
}

// P/D runs under the default cost routing; only its per-role soft bands need
// the ladder.
func TestPDAllowedUnderCostRouting(t *testing.T) {
	o := Options{CostRouting: true, PDPrefillBackends: []string{"http://p:8000"}, PDPrefillMaxNodeConcurrency: 8}
	if err := o.validateRouting(); err != nil {
		t.Errorf("--pd-prefill-backends under cost routing: %v", err)
	}
}

func TestValidatePDDynamicMode(t *testing.T) {
	routes := []Route{{Patterns: "*", Endpoints: []string{"http://p:8000", "http://d1:8000", "http://d2:8000"}}}
	floor := []string{"http://p:8000"}
	cases := []struct {
		name string
		o    Options
		want string
	}{
		{"split, reject mode", Options{Routes: routes, PrefillSplit: true}, ""},
		{"split, pd mode", Options{Routes: routes, PrefillSplit: true, PrefillSplitMode: PrefillSplitPD}, ""},
		{"pd mode with a floor", Options{Routes: routes, PrefillSplit: true, PrefillSplitMode: PrefillSplitPD, PDPrefillBackends: floor}, ""},
		{"pd mode band without a static set", Options{Routes: routes, PrefillSplit: true, PrefillSplitMode: PrefillSplitPD, PDPrefillMaxNodeConcurrency: 8}, ""},
		{"floor with reject mode", Options{Routes: routes, PrefillSplit: true, PDPrefillBackends: floor}, "unless --prefill-split-mode=pd"},
		{"pd mode without the split", Options{Routes: routes, PrefillSplitMode: PrefillSplitPD}, "needs --prefill-split"},
		{"unknown mode", Options{Routes: routes, PrefillSplit: true, PrefillSplitMode: "both"}, "want reject or pd"},
	}
	for _, c := range cases {
		err := c.o.validatePD()
		switch {
		case c.want == "" && err != nil:
			t.Errorf("%s: %v", c.name, err)
		case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
			t.Errorf("%s: err = %v, want %q", c.name, err, c.want)
		}
	}
}

// Both modes take the same split flags; pd mode replaces the reject mode's
// manager and runs its legs under the P flow's concurrency band.
func TestPDDynamicConfigReusesTheSplitFlags(t *testing.T) {
	split := Options{PrefillSplit: true, PrefillNodeBudgetTokens: 500_000, PrefillMaxFleetFraction: 0.4,
		PrefillConvertAfter: 20 * time.Second, PrefillReleaseIdle: 90 * time.Second, NodeConcurrency: 64}
	if split.pdDynamicConfig(false) != nil || split.prefillConfig() == nil {
		t.Fatal("reject mode must build the split's manager only")
	}
	o := split
	o.PrefillSplitMode, o.PDPrefillMaxNodeConcurrency, o.PrefillDrainDelay, o.PrefillMinNodes = PrefillSplitPD, 8, 2*time.Second, 2
	if o.prefillConfig() != nil {
		t.Error("pd mode also built the reject mode's manager")
	}
	if o.pdDynamicConfig(true) != nil {
		t.Error("pd mode manager for a transparent route")
	}
	c := o.pdDynamicConfig(false)
	if c.NodeBudgetTokens != 500_000 || c.MaxFleetFraction != 0.4 || c.ConvertAfter != 20*time.Second ||
		c.ReleaseIdle != 90*time.Second || c.NodeConcurrency != 8 || c.DrainDelay != 2*time.Second || c.MinPrefill != 2 {
		t.Errorf("config = %+v, want the split flags and the prefill band's concurrency 8", c)
	}
	if !strings.HasPrefix(o.signalsLog(), "prefill=[") {
		t.Errorf("signals log %q is not per role in pd mode", o.signalsLog())
	}
	// A budget left at 0 takes the pd default; the reject mode keeps the
	// manager's.
	o.PrefillNodeBudgetTokens, o.PrefillLoadWeight = 0, 0.02
	if c := o.pdDynamicConfig(false); c.NodeBudgetTokens != DefaultPDNodeBudgetTokens || c.LoadWeight != 0.02 {
		t.Errorf("pd budget %d load weight %v, want %d and 0.02", c.NodeBudgetTokens, c.LoadWeight, DefaultPDNodeBudgetTokens)
	}
	split.PrefillNodeBudgetTokens = 0
	if c := split.prefillConfig(); c.NodeBudgetTokens != 0 {
		t.Errorf("reject budget %d, want 0 (the manager's default)", c.NodeBudgetTokens)
	}
}
