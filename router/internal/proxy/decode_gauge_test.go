package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/weka/kvlab/router/internal/dialect/openai"
	"github.com/weka/kvlab/router/internal/registry"
)

func decodeBackend(t *testing.T, url string) *registry.Backend {
	t.Helper()
	b, err := registry.New(registry.Options{}).Add(registry.Spec{URL: url})
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	return b
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// The gauge counts a request from its first upstream body byte and returns to
// exactly 0 on success, upstream error and client cancel.
func TestDecodeTokensGaugeAcrossOutcomes(t *testing.T) {
	body := []byte(strings.Repeat("x", 4000)) // ~1000 estimated tokens

	run := func(t *testing.T, h http.HandlerFunc, cancelMidStream bool) *registry.Backend {
		t.Helper()
		up := httptest.NewServer(h)
		defer up.Close()
		b := decodeBackend(t, up.URL)
		p := New(DefaultConfig())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		req := httptest.NewRequest(http.MethodPost, "/v1/completions", nil).WithContext(ctx)
		rec := httptest.NewRecorder()
		done := make(chan struct{})
		go func() {
			defer close(done)
			p.Relay(rec, req, b, openai.New(), body, Auth{})
		}()
		if cancelMidStream {
			waitFor(t, "streaming to start", func() bool { return b.DecodeTokens() > 0 })
			cancel()
		}
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Fatal("Relay did not return")
		}
		return b
	}

	t.Run("success", func(t *testing.T) {
		release := make(chan struct{})
		var b *registry.Backend
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte("data: {}\n\n"))
			w.(http.Flusher).Flush()
			<-release
			w.Write([]byte("data: [DONE]\n\n"))
		}))
		defer up.Close()
		b = decodeBackend(t, up.URL)
		p := New(DefaultConfig())
		req := httptest.NewRequest(http.MethodPost, "/v1/completions", nil)
		done := make(chan struct{})
		go func() {
			defer close(done)
			p.Relay(httptest.NewRecorder(), req, b, openai.New(), body, Auth{})
		}()
		waitFor(t, "gauge to rise", func() bool { return b.DecodeTokens() == 1000 })
		close(release)
		<-done
		if got := b.DecodeTokens(); got != 0 {
			t.Fatalf("after success = %d, want 0", got)
		}
		if b.TTFTp90() != 0 { // 1 sample < min: reads 0, but the EWMA saw it
			t.Fatalf("p90 with one sample should read 0")
		}
		if b.TTFT() <= 0 {
			t.Fatalf("first byte should have fed the TTFT EWMA")
		}
	})

	t.Run("upstream error never counts", func(t *testing.T) {
		b := run(t, func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "boom", http.StatusInternalServerError)
		}, false)
		if got := b.DecodeTokens(); got != 0 {
			t.Fatalf("after 500 = %d, want 0", got)
		}
	})

	t.Run("client cancel mid-stream", func(t *testing.T) {
		b := run(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte("data: {}\n\n"))
			w.(http.Flusher).Flush()
			<-r.Context().Done()
		}, true)
		if got := b.DecodeTokens(); got != 0 {
			t.Fatalf("after cancel = %d, want 0", got)
		}
	})
}
