package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/weka/kvlab/llm"
)

// wireSeen is what one request looked like on the wire.
type wireSeen struct {
	body   map[string]any
	accept string
}

// TestNoStreamForcesOneAnswerOnTheWire sends a record that says stream=true
// and one that says stream=false, with and without --no-stream, and checks the
// body, the Accept header and the consumer (which a server answering in the
// requested format makes observable through the parsed usage) all agree.
func TestNoStreamForcesOneAnswerOnTheWire(t *testing.T) {
	var mu sync.Mutex
	var seen []wireSeen
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		mu.Lock()
		seen = append(seen, wireSeen{body: body, accept: r.Header.Get("Accept")})
		mu.Unlock()
		if s, _ := body["stream"].(bool); s {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
			fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":5}}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"hi"}}],`+
			`"usage":{"prompt_tokens":30,"completion_tokens":5}}`)
	}))
	defer ts.Close()

	for _, tc := range []struct {
		name       string
		noStream   bool
		recStream  bool
		wantStream bool
	}{
		{"flag off keeps a streaming record", false, true, true},
		{"flag off keeps a non-streaming record", false, false, false},
		{"flag on forces a streaming record off", true, true, false},
		{"flag on keeps a non-streaming record off", true, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mu.Lock()
			seen = nil
			mu.Unlock()
			p, err := newReplayPoster(fmt.Sprintf("dynamic/%s,type=openai,model=m", ts.URL),
				llm.APIKeys{OpenAI: "k"}, "", "", false, 0, 0, 0, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			p.noStream = tc.noStream
			req := RouterReplayRequest{
				Stream: tc.recStream, OutputTokens: 100,
				Messages: []RouterReplayMessage{userText("a", 40)},
			}
			m := p.do(context.Background(), req, "docs ", 1, "s1", "i1", 1,
				&autoState{stream: newCompletionStream(200)}, nil)
			if m.Error != nil {
				t.Fatalf("request failed: %v", m.Error)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(seen) != 1 {
				t.Fatalf("server saw %d requests, want 1", len(seen))
			}
			got := seen[0]
			if s, _ := got.body["stream"].(bool); s != tc.wantStream {
				t.Errorf("body stream = %v, want %v", s, tc.wantStream)
			}
			if _, has := got.body["stream_options"]; has != tc.wantStream {
				t.Errorf("stream_options present = %v, want %v", has, tc.wantStream)
			}
			wantAccept := "application/json"
			if tc.wantStream {
				wantAccept = "text/event-stream"
			}
			if got.accept != wantAccept {
				t.Errorf("Accept = %q, want %q", got.accept, wantAccept)
			}
			// Either consumer parses the matching response; a mismatched one
			// would leave the usage unobserved.
			if !m.UsageObserved || m.UsageData.OutputTokens.Count != 5 {
				t.Errorf("response was not parsed by the matching consumer: %+v", m.UsageData)
			}
		})
	}
}

func TestRunParamsRecordsNoStream(t *testing.T) {
	if !buildRunParams(AutoBenchmarkConfig{NoStream: true}, time.Now()).NoStream {
		t.Error("run_params must record --no-stream")
	}
	if buildRunParams(AutoBenchmarkConfig{}, time.Now()).NoStream {
		t.Error("run_params must not claim --no-stream when off")
	}
}
