package cli

import (
	"strings"
	"testing"

	"github.com/jessevdk/go-flags"

	"github.com/weka/wekai/router/serve"
)

func parseRouter(t *testing.T, args ...string) RouterServeCommand {
	t.Helper()
	var c RouterServeCommand
	p := flags.NewParser(&c, flags.Default)
	if _, err := p.ParseArgs(append([]string{"--backends", "http://a:8000"}, args...)); err != nil {
		t.Fatalf("parse %v: %v", args, err)
	}
	return c
}

// Cost routing is the default. The serve.Options field is true exactly when
// --no-cost-routing is absent; --cost-routing is a kept no-op.
func TestCostRoutingIsTheDefault(t *testing.T) {
	cases := []struct {
		args    []string
		on      bool
		wantErr bool
	}{
		{nil, true, false},
		{[]string{"--cost-routing"}, true, false},
		{[]string{"--no-cost-routing"}, false, false},
		{[]string{"--cost-routing", "--no-cost-routing"}, false, true},
	}
	for _, tc := range cases {
		c := parseRouter(t, tc.args...)
		if err := c.validateRouting(); (err != nil) != tc.wantErr {
			t.Errorf("%v: validateRouting err = %v, wantErr %v", tc.args, err, tc.wantErr)
		}
		if tc.wantErr {
			continue
		}
		if got := !c.NoCostRouting; got != tc.on {
			t.Errorf("%v: cost routing = %v, want %v", tc.args, got, tc.on)
		}
	}
	if err := (&RouterServeCommand{CostRouting: true, NoCostRouting: true}).validateRouting(); err == nil ||
		!strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("both flags: err = %v", err)
	}
}

// The go-flags default tag is a literal; it must equal the value the
// validation treats as "not set".
func TestSplitGuardTagMatchesServeDefault(t *testing.T) {
	if c := parseRouter(t); c.SplitGuard != serve.DefaultSplitGuard {
		t.Errorf("--cache-split-guard default = %v, serve.DefaultSplitGuard = %v", c.SplitGuard, serve.DefaultSplitGuard)
	}
}
