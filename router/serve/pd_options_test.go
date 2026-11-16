package serve

import (
	"strings"
	"testing"
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

func TestValidatePDElastic(t *testing.T) {
	routes := []Route{{Patterns: "*", Endpoints: []string{"http://p:8000", "http://d1:8000", "http://d2:8000"}}}
	cases := []struct {
		name string
		o    Options
		want string
	}{
		{"elastic alone", Options{Routes: routes, PDElastic: true, PDPrefillMin: 1}, ""},
		{"elastic with a floor", Options{Routes: routes, PDElastic: true, PDPrefillBackends: []string{"http://p:8000"}}, ""},
		{"elastic band without a static set", Options{Routes: routes, PDElastic: true, PDPrefillMaxNodeConcurrency: 8}, ""},
		{"min without elastic", Options{Routes: routes, PDPrefillMin: 2}, "needs --pd-elastic"},
		{"negative min", Options{Routes: routes, PDElastic: true, PDPrefillMin: -1}, "must not be negative"},
		{"release-below out of range", Options{Routes: routes, PDElastic: true, PrefillReleaseBelow: 1}, "[0, 1)"},
		{"with prefill split", Options{Routes: routes, PDElastic: true, PrefillSplit: true}, "mutually exclusive"},
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

func TestPDElasticConfigDefaultsAndRoles(t *testing.T) {
	if (Options{}).pdElasticConfig(false) != nil {
		t.Error("elastic config without --pd-elastic")
	}
	o := Options{PDElastic: true, PDPrefillMin: 2, PrefillReleaseBelow: 0.4, NodeConcurrency: 64, PDPrefillMaxNodeConcurrency: 8}
	if o.pdElasticConfig(true) != nil {
		t.Error("elastic config for a transparent route")
	}
	c := o.pdElasticConfig(false)
	if c.ConvertAfter != DefaultPDConvertAfter || c.ReleaseIdle != DefaultPDReleaseIdle {
		t.Errorf("hold times %v/%v, want the elastic defaults", c.ConvertAfter, c.ReleaseIdle)
	}
	if c.MinPrefill != 2 || c.ReleaseBelow != 0.4 || c.NodeConcurrency != 8 {
		t.Errorf("config = %+v, want min 2, release-below 0.4, the prefill band's concurrency 8", c)
	}
	if !strings.HasPrefix(o.signalsLog(), "prefill=[") {
		t.Errorf("signals log %q is not per role under --pd-elastic", o.signalsLog())
	}
}
