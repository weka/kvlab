package mockvllm

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"
)

// A P/D decode leg pulls its prompt's KV from the prefill node instead of
// recomputing it: on a cold engine with a slow prefill rate it answers at
// once, and counts the prompt as external_kv_transfer. The prefill leg on its
// own engine pays the recompute.
func TestDecodeLegWithRemotePrefillSkipsPromptCompute(t *testing.T) {
	cfg := Config{DefaultMaxTokens: 1, ColdInputTPS: 20_000}
	prompt := strings.Repeat("all work and no play ", 2_000) // ~10k tokens: 0.5s cold

	post := func(url string, kv any) time.Duration {
		t.Helper()
		body := map[string]any{"model": "m", "max_tokens": 1,
			"messages": []map[string]any{{"role": "user", "content": prompt}}}
		if kv != nil {
			body["kv_transfer_params"] = kv
		}
		b, _ := json.Marshal(body)
		start := time.Now()
		resp, err := http.Post(url+"/v1/chat/completions", "application/json", bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status %d", resp.StatusCode)
		}
		return time.Since(start)
	}

	pTS, _ := newTestServer(t, cfg)
	if d := post(pTS.URL, map[string]any{"do_remote_decode": true}); d < 300*time.Millisecond {
		t.Errorf("prefill leg took %v, want the cold recompute (~500ms)", d)
	}
	dTS, dSrv := newTestServer(t, cfg)
	if d := post(dTS.URL, map[string]any{"do_remote_prefill": true, "remote_engine_id": "p"}); d > 200*time.Millisecond {
		t.Errorf("decode leg took %v, want no prompt compute", d)
	}
	st := dSrv.Engine().Stats()
	if st.ExternalTokens == 0 || st.ExternalTokens != st.PromptTokens {
		t.Errorf("decode leg external tokens = %d of %d prompt tokens, want all", st.ExternalTokens, st.PromptTokens)
	}
}
