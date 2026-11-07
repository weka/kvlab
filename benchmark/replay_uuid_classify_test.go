package benchmark

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func stampSet(pairs ...[2]string) map[string]turnStamp {
	m := map[string]turnStamp{}
	for i, p := range pairs {
		m[p[0]] = turnStamp{Idx: i, UUID: p[1], Label: p[0]}
	}
	return m
}

// TestClassifyRecite pins the attribution, because the whole value of the
// breakdown is that a reader can trust which bucket a miss landed in. A miss
// filed as "no evidence" when the response plainly carried another marker from
// the same prompt would understate how intact the content was — the exact
// direction that turns a prompt defect into a reported cache failure.
func TestClassifyRecite(t *testing.T) {
	const (
		a = "11111111-1111-4111-8111-111111111111" // asked
		b = "22222222-2222-4222-8222-222222222222" // asked
		c = "33333333-3333-4333-8333-333333333333" // in the prompt, not asked
	)
	inj := &uuidInjection{
		StampByHash:  stampSet([2]string{"turn-1", a}, [2]string{"turn-2", b}, [2]string{"turn-9", c}),
		ReciteUUIDs:  []string{a, b},
		ReciteLabels: []string{"turn-1", "turn-2"},
	}

	t.Run("wrong turn from this prompt is a substitution", func(t *testing.T) {
		// The model answered with a marker it could only have read off a tag
		// in this very request, so the content was there.
		got := classifyRecite(a+", "+c, inj, []bool{true, false})
		if got.Substituted != 1 || got.Absent != 0 {
			t.Errorf("got %+v, want 1 substituted / 0 absent", got)
		}
		if got.NoIDs || got.EchoedTags {
			t.Errorf("got %+v, want no ask-quality flags on a response full of guids", got)
		}
	})

	t.Run("nothing produced is unattributed", func(t *testing.T) {
		got := classifyRecite("I am afraid I cannot help with that.", inj, []bool{false, false})
		if got.Absent != 2 || got.Substituted != 0 {
			t.Errorf("got %+v, want 2 absent / 0 substituted", got)
		}
		if !got.NoIDs {
			t.Error("a response with no guid at all must raise NoIDs; it measures the ask, not the fleet")
		}
	})

	t.Run("echoed tag names is an ask defect", func(t *testing.T) {
		got := classifyRecite("[turn-1], [turn-2]", inj, []bool{false, false})
		if !got.EchoedTags || !got.NoIDs {
			t.Errorf("got %+v, want EchoedTags and NoIDs: the response repeated the instruction's "+
				"list instead of resolving it", got)
		}
	})

	t.Run("guids plus a turn mention is annotation, not echo", func(t *testing.T) {
		// A response that answers correctly AND names the turn is being
		// helpful. Counting it as an echo would inflate the ask-defect signal
		// precisely when the ask is working.
		got := classifyRecite("[turn-1] "+a+", [turn-2] "+b, inj, []bool{true, true})
		if got.EchoedTags || got.NoIDs {
			t.Errorf("got %+v, want no ask-quality flags", got)
		}
	})

	t.Run("all found attributes nothing", func(t *testing.T) {
		if got := classifyRecite(a+", "+b, inj, []bool{true, true}); got.Substituted+got.Absent != 0 {
			t.Errorf("got %+v, want no misses attributed", got)
		}
	})

	t.Run("nil injection is inert", func(t *testing.T) {
		if got := classifyRecite("anything", nil, nil); got != (reciteOutcome{}) {
			t.Errorf("got %+v, want zero value", got)
		}
	})
}

// TestClassifyReciteAgreesWithPresence: substitution is decided with the same
// substring test presence uses. If they diverged, a marker could be scored
// found by one and substituted-for by the other, and the buckets would not sum
// to the miss count they are supposed to explain.
func TestClassifyReciteAgreesWithPresence(t *testing.T) {
	const a = "44444444-4444-4444-4444-444444444444"
	inj := &uuidInjection{
		StampByHash: stampSet([2]string{"turn-1", a}),
		ReciteUUIDs: []string{a},
	}
	resp := "the id is " + a + " as recorded"
	found := []bool{containsUUID(resp, a)}
	if !found[0] {
		t.Fatal("containsUUID disagrees with a plain substring match")
	}
	if got := classifyRecite(resp, inj, found); got.Substituted+got.Absent != 0 {
		t.Errorf("got %+v, want nothing attributed when the marker was found", got)
	}
}

func TestFindStraysReportsTurnAndChannel(t *testing.T) {
	const (
		a = "11111111-1111-4111-8111-111111111111"
		c = "33333333-3333-4333-8333-333333333333"
		d = "44444444-4444-4444-8444-444444444444"
	)
	inj := &uuidInjection{
		StampByHash: map[string]turnStamp{
			"h1": {Idx: 0, UUID: a, Label: "turn-1"},
			"h9": {Idx: 8, UUID: c, Label: "turn-9"},
			"h5": {Idx: 4, UUID: d, Label: "turn-5"},
		},
		ReciteUUIDs: []string{a},
	}
	got := findStrays("think "+c+" ans "+d+" "+d, d, c, inj)
	if len(got) != 2 {
		t.Fatalf("got %+v, want 2 strays", got)
	}
	// ordered by turn index: d (4) before c (8)
	if got[0].UUID != d || got[0].Idx != 4 || got[0].Where != "content" {
		t.Errorf("got[0]=%+v", got[0])
	}
	if got[1].UUID != c || got[1].Idx != 8 || got[1].Where != "reasoning" {
		t.Errorf("got[1]=%+v", got[1])
	}
	if s := findStrays(c, "", "", inj); len(s) != 1 || s[0].Where != "unsplit" {
		t.Errorf("unsplit: %+v", s)
	}
	if s := findStrays(a, a, "", inj); len(s) != 0 {
		t.Errorf("asked marker must not be a stray: %+v", s)
	}
	if findStrays(c, "", "", nil) != nil {
		t.Error("nil injection must yield nil")
	}
}

func TestMissDumpKeepsOnlyMisses(t *testing.T) {
	dir := t.TempDir()
	d, err := newRequestDumper(dumpMisses, dir, 10)
	if err != nil {
		t.Fatal(err)
	}
	cases := []dumpMeta{
		{Series: 1, Instance: "i", Turn: 0, Expected: []string{"x"}, Found: []bool{true}},
		{Series: 1, Instance: "i", Turn: 1, Expected: []string{"x"}, Found: []bool{false}, MissAbsent: 1},
		{Series: 1, Instance: "i", Turn: 2, Expected: []string{"x"}, Found: []bool{false}, MissSubstituted: 1,
			Strays: []strayMarker{{UUID: "u", Idx: 3, Label: "turn-4", Where: "reasoning"}}},
	}
	for _, m := range cases {
		d.dump(m, []byte("req"), []byte("resp"), []byte("merged"))
	}
	if got, _ := filepath.Glob(filepath.Join(dir, "*.meta.json")); len(got) != 2 {
		t.Fatalf("miss mode wrote %v, want turns 1 and 2", got)
	}
	d2dir := t.TempDir()
	d2, _ := newRequestDumper(dumpMisses, d2dir, 10)
	d2.onlySubstituted = true
	for _, m := range cases {
		d2.dump(m, []byte("req"), []byte("resp"), []byte("merged"))
	}
	got, _ := filepath.Glob(filepath.Join(d2dir, "*.meta.json"))
	if len(got) != 1 || !strings.Contains(got[0], "t002") {
		t.Fatalf("substituted-only wrote %v, want only t002", got)
	}
	b, _ := os.ReadFile(got[0])
	if !strings.Contains(string(b), `"stray_markers"`) || !strings.Contains(string(b), `"reasoning"`) {
		t.Errorf("meta lacks stray details: %s", b)
	}
}
