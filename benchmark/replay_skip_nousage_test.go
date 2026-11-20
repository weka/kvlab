package benchmark

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// arrivalServer answers at once and records when each tagged request arrived.
func arrivalServer(t *testing.T) (*httptest.Server, func() map[string]time.Time) {
	var mu sync.Mutex
	arrivals := map[string]time.Time{}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		// Requests are told apart by their user-message count (content is
		// generated from the captured hashes, not the hashes themselves).
		tag := map[int]string{1: "turn-a", 2: "turn-b", 3: "turn-c"}[strings.Count(string(body), `"role":"user"`)]
		mu.Lock()
		arrivals[tag] = time.Now()
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`)
	}))
	t.Cleanup(ts.Close)
	return ts, func() map[string]time.Time {
		mu.Lock()
		defer mu.Unlock()
		out := map[string]time.Time{}
		for k, v := range arrivals {
			out[k] = v
		}
		return out
	}
}

func skipCfg(ts *httptest.Server, send bool) AutoBenchmarkConfig {
	return AutoBenchmarkConfig{
		Model:             fmt.Sprintf("dynamic/%s,type=openai,model=test-model", ts.URL),
		ReplayNoStamp:     true,
		ReplayRealtime:    true,
		replayThink:       true,
		ReplaySendNoUsage: send,
	}
}

// Three turns timed think-from-prev; the middle one has no captured usage.
// It is not sent, and the third keeps its captured offset: the middle turn's
// think (100ms) and duration (300ms), then its own think (200ms).
func TestReplaySkipsNoUsageAndKeepsTheNextTurnsTiming(t *testing.T) {
	for _, send := range []bool{false, true} {
		ts, arrivals := arrivalServer(t)
		t100, t200 := int64(100), int64(200)
		sess := RouterReplaySession{SessionID: "s", StartTs: "2026-05-12T08:50:06Z",
			Instances: []RouterReplayInstance{{InstanceID: "i1", Role: "main", Requests: []RouterReplayRequest{
				{RequestID: 1, Ts: "2026-05-12T08:50:06Z", TotalMs: 10, OutputTokens: 10,
					Messages: []RouterReplayMessage{userText("turn-a", 40)}},
				{RequestID: 2, Ts: "2026-05-12T08:50:06.110Z", TotalMs: 300, OutputTokens: 0, MaxTokens: 128000,
					ThinkMs: &t100, ThinkRef: ThinkRefPrev,
					Messages: []RouterReplayMessage{userText("turn-a", 40), userText("turn-b", 40)}},
				{RequestID: 3, Ts: "2026-05-12T08:50:06.610Z", TotalMs: 10, OutputTokens: 10,
					ThinkMs: &t200, ThinkRef: ThinkRefPrev,
					Messages: []RouterReplayMessage{userText("turn-a", 40), userText("turn-b", 40), userText("turn-c", 40)}},
			}}}}
		st := newTestAutoState(4)
		runRouterReplaySession(context.Background(), skipCfg(ts, send), st, nil, sess, 1,
			endpointPicker{}, 30*time.Second, strings.Repeat("docs ", 100), newConcurrencyGate(4, false))
		got := arrivals()
		if send {
			if _, ok := got["turn-b"]; !ok || st.skippedNoUsage.Load() != 0 {
				t.Errorf("--replay-skip-no-usage=false: turn b sent %v, skipped %d", ok, st.skippedNoUsage.Load())
			}
			continue
		}
		if _, ok := got["turn-b"]; ok {
			t.Fatal("the request with no captured usage was sent")
		}
		if st.skippedNoUsage.Load() != 1 || st.totalCompleted.Load() != 2 {
			t.Errorf("skipped %d completed %d, want 1 and 2", st.skippedNoUsage.Load(), st.totalCompleted.Load())
		}
		gap := got["turn-c"].Sub(got["turn-a"])
		if gap < 580*time.Millisecond || gap > 900*time.Millisecond {
			t.Errorf("turn c arrived %v after turn a, want ~600ms (100ms think + 300ms skipped turn + 200ms think)", gap)
		}
	}
}

// A skipped request that spawned a sub-agent still releases it, timed from
// the skipped request's captured end.
func TestReplaySkippedSpawnParentReleasesItsChild(t *testing.T) {
	ts, arrivals := arrivalServer(t)
	t200 := int64(200)
	sess := RouterReplaySession{SessionID: "s", StartTs: "2026-05-12T08:50:06Z",
		Instances: []RouterReplayInstance{
			{InstanceID: "p", Role: "main", Requests: []RouterReplayRequest{
				{RequestID: 1, Ts: "2026-05-12T08:50:06Z", TotalMs: 300, OutputTokens: 0,
					Messages: []RouterReplayMessage{userText("parent-x", 40)}}}}, // tagged turn-a
			{InstanceID: "c", Role: "sub-agent", ParentSpawnRequestID: 1, Requests: []RouterReplayRequest{
				{RequestID: 2, Ts: "2026-05-12T08:50:06.500Z", TotalMs: 10, OutputTokens: 10,
					ThinkMs: &t200, ThinkRef: ThinkRefSpawn,
					Messages: []RouterReplayMessage{userText("child-x", 40), userText("child-y", 40)}}}}, // tagged turn-b
		}}
	st := newTestAutoState(4)
	start := time.Now()
	done := make(chan struct{})
	go func() {
		runRouterReplaySession(context.Background(), skipCfg(ts, false), st, nil, sess, 1,
			endpointPicker{}, 30*time.Second, strings.Repeat("docs ", 100), newConcurrencyGate(4, false))
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the child never ran: the skipped parent did not release it")
	}
	got := arrivals()
	if _, ok := got["turn-a"]; ok {
		t.Error("the parent with no captured usage was sent")
	}
	c, ok := got["turn-b"]
	if !ok {
		t.Fatal("the child was not sent")
	}
	if gap := c.Sub(start); gap < 480*time.Millisecond {
		t.Errorf("child sent %v after start, want ~500ms (300ms skipped parent + 200ms think)", gap)
	}
}

func TestCountNoUsageRequests(t *testing.T) {
	path := filepath.Join(t.TempDir(), "r.jsonl")
	body := `{"_schema":"replay-v5","summary":{}}` + "\n" +
		`{"session_id":"a","instances":[{"requests":[{"output_tokens":5},{"output_tokens":0}]}]}` + "\n" +
		`{"session_id":"b","instances":[{"requests":[{"output_tokens":0}]},{"requests":[{"output_tokens":7}]}]}` + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	n, total, err := countNoUsageRequests(path)
	if err != nil || n != 2 || total != 4 {
		t.Errorf("count %d of %d, %v; want 2 of 4", n, total, err)
	}
	line := renderModelOneLiner(&displaySnapshot{skippedNoUsage: 3})
	if !strings.Contains(line, " skipped=3") {
		t.Errorf("progress line %q lacks skipped=3", line)
	}
}
