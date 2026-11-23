package cli

import (
	"strings"
	"testing"
)

func TestLateSystemFlagsMutuallyExclusive(t *testing.T) {
	both := &BenchmarkAutoOptions{ReplaySystemFirst: true, ReplayDropLateSystem: true}
	if err := both.validateLateSystem(); err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("both flags: err = %v, want mutually exclusive", err)
	}
	for _, c := range []*BenchmarkAutoOptions{{}, {ReplaySystemFirst: true}, {ReplayDropLateSystem: true}} {
		if err := c.validateLateSystem(); err != nil {
			t.Fatalf("%+v: unexpected err %v", *c, err)
		}
	}
}
