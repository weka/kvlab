package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/weka/kvlab/router/internal/dialect/openai"
	"github.com/weka/kvlab/router/internal/policy"
)

// The prefill charge is held until the first upstream body byte and is
// returned to zero by the time the request ends.
func TestPrefillTokensGaugeHeldUntilFirstByte(t *testing.T) {
	body := []byte(strings.Repeat("x", 4000))
	started := make(chan struct{})
	release := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		close(started)
		<-release
		w.Write([]byte("data: {}\n\n"))
	}))
	defer up.Close()
	b := decodeBackend(t, up.URL)
	other := decodeBackend(t, "http://other:8000")
	p := New(DefaultConfig())
	rr := &policy.RoutingRequest{Prefill: &policy.PrefillEstimate{Backend: b, Tokens: 250}}
	if got := prefillCharge(rr, b, body); got != 250 {
		t.Fatalf("charge on selected backend = %d, want 250", got)
	}
	if got := prefillCharge(rr, other, body); got != 1000 {
		t.Fatalf("charge on other backend = %d, want full 1000", got)
	}
	if got := prefillCharge(&policy.RoutingRequest{}, b, body); got != 0 {
		t.Fatalf("charge with cost routing off = %d, want 0", got)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		p.attempt(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/v1/completions", nil),
			b, openai.New(), body, 250, new(bool), false, nil, nil, Auth{})
	}()
	<-started
	if got := b.PrefillTokens(); got != 250 {
		t.Fatalf("prefill before first byte = %d, want 250", got)
	}
	close(release)
	waitFor(t, "prefill to clear", func() bool { return b.PrefillTokens() == 0 })
	<-done
	if got := b.PrefillTokens(); got != 0 {
		t.Fatalf("prefill after request = %d, want 0", got)
	}
}
