package pd

import (
	"testing"

	"github.com/weka/kvlab/kvcache"
	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/policy/affinity"
)

func TestStoredRecordsPromptsAndIsNilSafe(t *testing.T) {
	var none *Stored
	none.Record(&policy.RoutingRequest{})
	if none.Depth(&policy.RoutingRequest{}) != 0 {
		t.Fatal("a nil record holds something")
	}
	s, err := NewStored(affinity.Config{PoolName: t.Name()})
	if err != nil {
		t.Fatal(err)
	}
	turn1 := &policy.RoutingRequest{Model: "m", Units: []kvcache.Unit{{Hash: 1, Tokens: 300}, {Hash: 2, Tokens: 300}}}
	turn2 := &policy.RoutingRequest{Model: "m", Units: append(append([]kvcache.Unit{}, turn1.Units...), kvcache.Unit{Hash: 3, Tokens: 50})}
	if s.Depth(turn2) != 0 {
		t.Fatal("nothing recorded yet")
	}
	s.Record(turn1)
	if got := s.Depth(turn2); got != 600 {
		t.Errorf("depth = %d, want the 600 tokens of turn 1", got)
	}
}
