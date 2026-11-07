package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// interleavedStreamPoster serves a stream whose first chunk carries BOTH
// content (the first half of the recited UUID) and reasoning, so the merged
// Response splits the UUID while ContentOnly keeps it intact.
func runInterleavedRecite(t *testing.T) RequestMetrics {
	t.Helper()
	const stamp = "perchan-run"
	su := buildSessionUUIDs(RouterReplaySession{
		SessionID: "s",
		Instances: []RouterReplayInstance{{
			Requests: []RouterReplayRequest{{Messages: []RouterReplayMessage{userText("turn-a", 40)}}},
		}},
	}, stamp)
	u := su.uuids[0]

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		chunk := func(content, reasoning string) {
			d := map[string]any{"content": content}
			if reasoning != "" {
				d["reasoning"] = reasoning
			}
			b, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": d}}})
			fmt.Fprintf(w, "data: %s\n\n", b)
		}
		chunk(u[:10], "hmm")
		chunk(u[10:], "")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer ts.Close()

	reg := newUUIDRegistry()
	reg.Acquire(su.uuids, uuidHolder{Series: 1})
	p := newContamTestPoster(t, ts.URL, reg, true)
	st := &autoState{stream: newCompletionStream(200)}
	req := RouterReplayRequest{
		Stream: true, OutputTokens: 400,
		Messages: []RouterReplayMessage{userText("turn-a", 40)},
	}
	m := p.do(context.Background(), req, strings.Repeat("docs ", 60), 3, "s", "i1", 1, st, su)
	if strings.Contains(m.Response, u) {
		t.Fatalf("fixture is not interleaved: merged Response %q contains the UUID", m.Response)
	}
	if !strings.Contains(m.ContentOnly, u) {
		t.Fatalf("fixture broken: ContentOnly %q lacks the UUID", m.ContentOnly)
	}
	if len(m.UUIDFound) == 0 {
		t.Fatal("no presence scoring happened")
	}
	return m
}

func TestPresenceRecoversInterleavedUUID(t *testing.T) {
	m := runInterleavedRecite(t)
	if !m.UUIDFound[0] {
		t.Errorf("UUIDFound[0] = false; the UUID is intact in ContentOnly")
	}
	if m.UUIDRecoveredPerChannel != 1 {
		t.Errorf("recovered = %d, want 1", m.UUIDRecoveredPerChannel)
	}
	if m.MissAbsent+m.MissSubstituted != 0 {
		t.Errorf("recovered UUID still attributed as a miss: %+v", m)
	}
}
