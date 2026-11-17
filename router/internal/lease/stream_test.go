package lease

import (
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/registry"
)

// A backend knows when its oldest in-flight request started, and forgets each
// request when its lease is released, however many times that happens.
func TestOldestStreamFollowsLeases(t *testing.T) {
	reg := registry.New(registry.Options{})
	b, _ := reg.Add(registry.Spec{URL: "http://s", Prov: registry.ProvStatic, Capacity: 1})
	if !b.OldestStream().IsZero() {
		t.Fatal("oldest stream set with nothing in flight")
	}
	first := Acquire(b)
	time.Sleep(5 * time.Millisecond)
	second := Acquire(b)
	oldest := b.OldestStream()
	if oldest.IsZero() || time.Since(oldest) < 5*time.Millisecond {
		t.Fatalf("oldest stream %v, want the first lease's start", oldest)
	}
	first.Release()
	first.Release()
	if !b.OldestStream().After(oldest) {
		t.Error("oldest stream did not move to the second lease after the first ended")
	}
	second.Release()
	if !b.OldestStream().IsZero() {
		t.Error("oldest stream set after every lease ended")
	}
}
