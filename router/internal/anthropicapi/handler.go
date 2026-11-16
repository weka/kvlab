package anthropicapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/weka/kvlab/kvcache"
)

const (
	messagesPath    = "/v1/messages"
	countTokensPath = "/v1/messages/count_tokens"
	chatPath        = "/v1/chat/completions"
	tokenizePath    = "/tokenize"
)

// Middleware serves the Messages API by translation, passing everything else
// to next unchanged. next is the router's ordinary handler; a translated
// request reaches it as a /v1/chat/completions request in every respect.
func Middleware(cfg Config, next http.Handler) http.Handler {
	cfg.withDefaults()
	h := &handler{cfg: cfg, next: next}
	if cfg.DebugDir != "" {
		if err := os.MkdirAll(cfg.DebugDir, 0o755); err != nil {
			cfg.Log.Warn("anthropic debug dir unusable; exchanges will not be recorded",
				"dir", cfg.DebugDir, "err", err)
		} else {
			h.debug = &debugLog{path: filepath.Join(cfg.DebugDir, "anthropic-exchanges.jsonl")}
		}
	}
	return h
}

type handler struct {
	cfg   Config
	next  http.Handler
	debug *debugLog
}

// split returns the user prefix and the API path, and whether the path is one
// this handler serves.
func (h *handler) split(path string) (prefix, api string, ok bool) {
	for _, p := range []string{countTokensPath, messagesPath} {
		if path == p {
			return "", p, true
		}
		if h.cfg.UserPrefix && strings.HasSuffix(path, p) {
			pre := strings.TrimSuffix(path, p)
			// Exactly one leading segment: /<user>.
			if len(pre) > 1 && pre[0] == '/' && !strings.Contains(pre[1:], "/") {
				return pre, p, true
			}
		}
	}
	return "", "", false
}

func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	prefix, api, ok := h.split(r.URL.Path)
	if !ok {
		h.next.ServeHTTP(w, r)
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed; use POST")
		return
	}
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, h.cfg.MaxBodyBytes))
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body exceeds the configured limit")
			return
		}
		writeError(w, http.StatusBadRequest, "could not read request body")
		return
	}
	tr, err := toOpenAI(raw, &h.cfg)
	if err != nil {
		var re *RequestError
		if errors.As(err, &re) {
			writeError(w, http.StatusBadRequest, re.Msg)
			return
		}
		writeError(w, http.StatusInternalServerError, "could not translate request: "+err.Error())
		return
	}
	if len(tr.dropped) > 0 {
		h.cfg.Log.Debug("anthropic request carried input the backend cannot represent; dropped",
			"dropped", strings.Join(dedupe(tr.dropped), ", "))
	}
	if api == countTokensPath {
		h.countTokens(w, r, prefix, tr)
		return
	}
	h.messages(w, r, prefix, tr, raw)
}

// backendRequest is the translated request as the router's handler sees it.
func backendRequest(r *http.Request, path string, body []byte) *http.Request {
	out := r.Clone(r.Context())
	out.URL.Path = path
	out.URL.RawPath = ""
	out.RequestURI = ""
	out.Body = io.NopCloser(bytes.NewReader(body))
	out.ContentLength = int64(len(body))
	out.Header.Set("Content-Length", strconv.Itoa(len(body)))
	out.Header.Set("Content-Type", "application/json")
	// The response is read here, so it must arrive uncompressed. Go's
	// transport negotiates and undoes its own gzip when the caller sets none.
	out.Header.Del("Accept-Encoding")
	for k := range out.Header {
		if strings.HasPrefix(strings.ToLower(k), "anthropic-") {
			out.Header.Del(k)
		}
	}
	return out
}

func (h *handler) messages(w http.ResponseWriter, r *http.Request, prefix string, tr *translated, raw []byte) {
	rw := &respWriter{client: w, tr: tr, cfg: &h.cfg, debug: h.debug != nil}
	h.next.ServeHTTP(rw, backendRequest(r, prefix+chatPath, tr.body))
	rw.finish(r.Context().Err() != nil)
	if h.debug != nil {
		h.debug.record(raw, tr.body, rw.status, rw.upstream.Bytes(), rw.sent.Bytes(), h.cfg.Log)
	}
}

// countTokens answers /v1/messages/count_tokens. The backend's own /tokenize
// is exact when the pool exposes it; otherwise the count is the router's usual
// byte-based estimate of the translated prompt, which is what Claude Code uses
// this for anyway (a context-size gauge, not billing).
func (h *handler) countTokens(w http.ResponseWriter, r *http.Request, prefix string, tr *translated) {
	var oa map[string]any
	_ = json.Unmarshal(tr.body, &oa)
	tok := map[string]any{"model": oa["model"], "messages": oa["messages"],
		"add_generation_prompt": true}
	if t, ok := oa["tools"]; ok {
		tok["tools"] = t
	}
	body, _ := json.Marshal(tok)
	rec := &bufferWriter{hdr: http.Header{}}
	h.next.ServeHTTP(rec, backendRequest(r, prefix+tokenizePath, body))
	n := 0
	if rec.status == http.StatusOK {
		var resp struct {
			Count int `json:"count"`
		}
		if json.Unmarshal(rec.buf.Bytes(), &resp) == nil && resp.Count > 0 {
			n = resp.Count
		}
	}
	if n == 0 {
		n = int(kvcache.EstimateTokens(len(tr.body)))
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]int{"input_tokens": n})
}

// bufferWriter captures a whole response.
type bufferWriter struct {
	hdr    http.Header
	status int
	buf    bytes.Buffer
}

func (b *bufferWriter) Header() http.Header { return b.hdr }
func (b *bufferWriter) WriteHeader(code int) {
	if b.status == 0 {
		b.status = code
	}
}
func (b *bufferWriter) Write(p []byte) (int, error) {
	if b.status == 0 {
		b.status = http.StatusOK
	}
	return b.buf.Write(p)
}

// respWriter receives the backend's Chat Completions response from the router
// and writes the Anthropic one to the client.
type respWriter struct {
	client http.ResponseWriter
	tr     *translated
	cfg    *Config
	hdr    http.Header

	status int
	mode   respMode
	buf    bytes.Buffer // JSON and error bodies, translated at the end
	conv   *streamConverter

	debug    bool
	upstream bytes.Buffer // what the backend sent, for the debug log
	sent     bytes.Buffer // what the client got, for the debug log
}

type respMode int

const (
	modeUnset respMode = iota
	modeSSE
	modeJSON
	modeError
)

// maxBufferedBody bounds a non-streamed body held for translation.
const maxBufferedBody = 64 << 20

func (rw *respWriter) Header() http.Header {
	if rw.hdr == nil {
		rw.hdr = http.Header{}
	}
	return rw.hdr
}

func (rw *respWriter) WriteHeader(code int) {
	if rw.status != 0 {
		return
	}
	rw.status = code
	ct := strings.ToLower(rw.Header().Get("Content-Type"))
	switch {
	case code != http.StatusOK:
		rw.mode = modeError
	case strings.HasPrefix(ct, "text/event-stream"):
		rw.mode = modeSSE
		rw.startSSE()
	default:
		rw.mode = modeJSON
	}
}

// copyHeaders passes on the headers that mean the same in both formats.
func (rw *respWriter) copyHeaders() {
	for _, k := range []string{"X-Request-Id", "Retry-After"} {
		if v := rw.Header().Get(k); v != "" {
			rw.client.Header().Set(k, v)
		}
	}
}

func (rw *respWriter) out() io.Writer {
	if rw.debug {
		return io.MultiWriter(rw.client, &rw.sent)
	}
	return rw.client
}

func (rw *respWriter) flushClient() {
	if f, ok := rw.client.(http.Flusher); ok {
		f.Flush()
	}
}

func (rw *respWriter) startSSE() {
	rw.copyHeaders()
	h := rw.client.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache")
	h.Del("Content-Length")
	rw.client.WriteHeader(http.StatusOK)
	warn := func(msg string, args ...any) { rw.cfg.Log.Warn("anthropic stream: "+msg, args...) }
	rw.conv = newStreamConverter(rw.out(), rw.flushClient, rw.tr, rw.cfg.Clock, rw.cfg.PingInterval, warn)
	rw.conv.start("", estimateInput(rw.tr))
}

func (rw *respWriter) Write(p []byte) (int, error) {
	if rw.status == 0 {
		rw.WriteHeader(http.StatusOK)
	}
	if rw.debug {
		rw.upstream.Write(p)
	}
	switch rw.mode {
	case modeSSE:
		return rw.conv.Write(p)
	default:
		if rw.buf.Len()+len(p) > maxBufferedBody {
			return 0, errors.New("anthropic translation: response body too large to buffer")
		}
		return rw.buf.Write(p)
	}
}

// Flush is a no-op for buffered modes; the stream converter flushes the
// client itself after each batch of events it writes.
func (rw *respWriter) Flush() {}

// finish writes whatever the backend's response translates to.
func (rw *respWriter) finish(clientGone bool) {
	switch rw.mode {
	case modeSSE:
		rw.conv.Close(clientGone)
	case modeJSON:
		rw.finishJSON()
	case modeError:
		if rw.status == 499 || clientGone {
			return // the client hung up; there is nobody to answer
		}
		msg := upstreamErrorMessage(rw.buf.Bytes())
		if msg == "" {
			msg = http.StatusText(rw.status)
		}
		rw.copyHeaders()
		rw.writeErr(rw.status, translateErrorMessage(rw.status, msg))
	case modeUnset:
		if !clientGone {
			rw.writeErr(http.StatusBadGateway, "the router produced no response")
		}
	}
}

func (rw *respWriter) writeErr(status int, msg string) {
	if rw.debug {
		rw.sent.Write(errorJSON(status, msg))
	}
	writeError(rw.client, status, msg)
}

func (rw *respWriter) finishJSON() {
	msg, warnings, err := fromOpenAI(rw.buf.Bytes(), rw.tr)
	if err != nil {
		rw.cfg.Log.Warn("anthropic translation: unusable backend response", "err", err,
			"body", truncate(rw.buf.String(), 500))
		rw.writeErr(http.StatusBadGateway, err.Error())
		return
	}
	for _, w := range warnings {
		rw.cfg.Log.Warn("anthropic translation: " + w)
	}
	rw.copyHeaders()
	if rw.tr.stream {
		// The client asked for a stream and the backend answered whole. Replay
		// the complete message as the event sequence the client expects.
		h := rw.client.Header()
		h.Set("Content-Type", "text/event-stream; charset=utf-8")
		h.Set("Cache-Control", "no-cache")
		rw.client.WriteHeader(http.StatusOK)
		replayAsStream(rw.out(), msg)
		rw.flushClient()
		return
	}
	b, _ := json.Marshal(msg)
	rw.client.Header().Set("Content-Type", "application/json")
	rw.client.WriteHeader(http.StatusOK)
	_, _ = rw.out().Write(b)
}

// replayAsStream renders a complete message as an Anthropic event stream.
func replayAsStream(w io.Writer, m *MessagesResponse) {
	ev := func(typ string, v any) {
		b, _ := json.Marshal(v)
		_, _ = io.WriteString(w, "event: "+typ+"\ndata: "+string(b)+"\n\n")
	}
	start := *m
	start.Content = []ResponseBlock{}
	start.StopReason, start.StopSequence = nil, nil
	ev("message_start", map[string]any{"type": "message_start", "message": start})
	for i, b := range m.Content {
		switch b.Type {
		case "text":
			ev("content_block_start", map[string]any{"type": "content_block_start", "index": i,
				"content_block": map[string]string{"type": "text", "text": ""}})
			ev("content_block_delta", map[string]any{"type": "content_block_delta", "index": i,
				"delta": map[string]string{"type": "text_delta", "text": *b.Text}})
		case "thinking":
			ev("content_block_start", map[string]any{"type": "content_block_start", "index": i,
				"content_block": map[string]string{"type": "thinking", "thinking": "", "signature": ""}})
			ev("content_block_delta", map[string]any{"type": "content_block_delta", "index": i,
				"delta": map[string]string{"type": "thinking_delta", "thinking": *b.Thinking}})
			ev("content_block_delta", map[string]any{"type": "content_block_delta", "index": i,
				"delta": map[string]string{"type": "signature_delta", "signature": ""}})
		case "tool_use":
			ev("content_block_start", map[string]any{"type": "content_block_start", "index": i,
				"content_block": map[string]any{"type": "tool_use", "id": b.ID, "name": b.Name,
					"input": map[string]any{}}})
			ev("content_block_delta", map[string]any{"type": "content_block_delta", "index": i,
				"delta": map[string]string{"type": "input_json_delta", "partial_json": string(b.Input)}})
		}
		ev("content_block_stop", map[string]any{"type": "content_block_stop", "index": i})
	}
	ev("message_delta", map[string]any{"type": "message_delta",
		"delta": map[string]any{"stop_reason": m.StopReason, "stop_sequence": m.StopSequence},
		"usage": m.Usage})
	ev("message_stop", map[string]string{"type": "message_stop"})
}

// estimateInput is message_start's input_tokens: the real count arrives only
// with the final usage, and message_delta then reports it.
func estimateInput(tr *translated) int {
	return int(kvcache.EstimateTokens(len(tr.body)))
}

func dedupe(in []string) []string {
	seen := map[string]bool{}
	out := in[:0:0]
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// debugLog appends one JSON line per exchange.
type debugLog struct {
	mu   sync.Mutex
	path string
}

func (d *debugLog) record(anthropicReq, openaiReq []byte, status int, upstream, sent []byte, log interface {
	Warn(string, ...any)
}) {
	rec := map[string]any{
		"anthropic_request":  json.RawMessage(anthropicReq),
		"openai_request":     json.RawMessage(openaiReq),
		"status":             status,
		"openai_response":    string(upstream),
		"anthropic_response": string(sent),
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	f, err := os.OpenFile(d.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		log.Warn("anthropic debug log", "err", err)
		return
	}
	defer f.Close()
	_, _ = f.Write(append(b, '\n'))
}
