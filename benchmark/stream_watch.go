package benchmark

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Error kinds recorded per request (requestDataRecord.ErrorKind) and counted
// on the progress line.
const (
	ErrorKindTTFT   = "ttft"   // no first token within --ttft-timeout
	ErrorKindIdle   = "idle"   // no new chunk for --stream-idle-timeout after the first token
	ErrorKindGlobal = "global" // --request-timeout, the overall safety net
	ErrorKindHTTP   = "http"   // a non-2xx response
	ErrorKindOther  = "other"
)

var errorKinds = [...]string{ErrorKindTTFT, ErrorKindIdle, ErrorKindGlobal, ErrorKindHTTP, ErrorKindOther}

// timeoutError is a request ended by one of the client's own deadlines.
type timeoutError struct {
	kind string
	msg  string
}

func (e *timeoutError) Error() string { return e.msg }

// errorKindOf classifies a request error. nil is "".
func errorKindOf(err error) string {
	if err == nil {
		return ""
	}
	var te *timeoutError
	if errors.As(err, &te) {
		return te.kind
	}
	if strings.HasPrefix(err.Error(), "status ") {
		return ErrorKindHTTP
	}
	return ErrorKindOther
}

// errorKindCounts counts errors per kind for the progress line.
type errorKindCounts [len(errorKinds)]atomic.Int64

func (c *errorKindCounts) add(kind string) {
	for i, k := range errorKinds {
		if k == kind {
			c[i].Add(1)
			return
		}
	}
	c[len(errorKinds)-1].Add(1)
}

// String is " (ttft=a idle=b global=c http=d other=e)", or "" with no error.
func (c *errorKindCounts) String() string {
	if c == nil {
		return ""
	}
	var parts []string
	total := int64(0)
	for i, k := range errorKinds {
		n := c[i].Load()
		total += n
		parts = append(parts, fmt.Sprintf("%s=%d", k, n))
	}
	if total == 0 {
		return ""
	}
	return " (" + strings.Join(parts, " ") + ")"
}

// streamWatch enforces the two stall deadlines of one streamed request: no
// first token within ttft, then no new chunk for idle. Either cancels the
// request's context with a *timeoutError cause. The consumers report tokens
// and chunks; a long stream that keeps producing is never cut by them.
// Methods are safe on nil.
type streamWatch struct {
	ttft, idle time.Duration
	start      time.Time
	cancel     context.CancelCauseFunc

	first  atomic.Bool
	lastNs atomic.Int64
	tokens atomic.Int64

	mu    sync.Mutex
	timer *time.Timer
	off   bool
}

type streamWatchKey struct{}

// withRequestDeadlines derives a request context bounded by the global
// timeout (cause: request timeout (global)) and, when ttft or idle is set, a
// streamWatch enforcing them. cancel releases everything.
func withRequestDeadlines(parent context.Context, global, ttft, idle time.Duration) (context.Context, context.CancelFunc) {
	ctx, cancelCause := context.WithCancelCause(parent)
	cancelGlobal := func() {}
	if global > 0 {
		ctx, cancelGlobal = context.WithTimeoutCause(ctx, global, globalCause{})
	}
	var w *streamWatch
	if ttft > 0 || idle > 0 {
		w = &streamWatch{ttft: ttft, idle: idle, start: time.Now(), cancel: cancelCause}
		if ttft > 0 {
			w.timer = time.AfterFunc(ttft, w.fire)
		}
		ctx = context.WithValue(ctx, streamWatchKey{}, w)
	}
	return ctx, func() {
		w.stop()
		cancelGlobal()
		cancelCause(context.Canceled)
	}
}

// globalCause marks the global deadline; the message is built where the
// request's phase is known (classifyDeadline).
type globalCause struct{}

func (globalCause) Error() string { return "request timeout (global)" }

func watchFrom(ctx context.Context) *streamWatch {
	w, _ := ctx.Value(streamWatchKey{}).(*streamWatch)
	return w
}

// fire runs when the armed deadline passes: before the first token it is the
// TTFT deadline; after it, the idle one unless a chunk arrived meanwhile, in
// which case it re-arms for the rest.
func (w *streamWatch) fire() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.off {
		return
	}
	if !w.first.Load() {
		w.off = true
		w.cancel(&timeoutError{kind: ErrorKindTTFT,
			msg: fmt.Sprintf("ttft timeout %s", fmtSecs(w.ttft))})
		return
	}
	if w.idle <= 0 {
		return
	}
	since := time.Duration(time.Now().UnixNano() - w.lastNs.Load())
	if since < w.idle {
		w.timer = time.AfterFunc(w.idle-since, w.fire)
		return
	}
	w.off = true
	w.cancel(&timeoutError{kind: ErrorKindIdle,
		msg: fmt.Sprintf("stream idle %s after %d tokens", fmtSecs(w.idle), w.tokens.Load())})
}

// token reports a streamed token (a content/reasoning delta).
func (w *streamWatch) token() {
	if w == nil {
		return
	}
	w.tokens.Add(1)
	w.lastNs.Store(time.Now().UnixNano())
	if w.first.CompareAndSwap(false, true) {
		// From here only the idle deadline applies.
		w.mu.Lock()
		if w.timer != nil {
			w.timer.Stop()
		}
		if !w.off && w.idle > 0 {
			w.timer = time.AfterFunc(w.idle, w.fire)
		}
		w.mu.Unlock()
	}
}

// chunk reports any other stream event; after the first token it counts as
// activity for the idle deadline.
func (w *streamWatch) chunk() {
	if w != nil && w.first.Load() {
		w.lastNs.Store(time.Now().UnixNano())
	}
}

// stop disarms the watch: the request finished, or is not streamed (a plain
// response has no tokens to watch; only the global deadline applies).
func (w *streamWatch) stop() {
	if w == nil {
		return
	}
	w.mu.Lock()
	w.off = true
	if w.timer != nil {
		w.timer.Stop()
	}
	w.mu.Unlock()
}

func fmtSecs(d time.Duration) string {
	return fmt.Sprintf("%gs", d.Round(time.Millisecond).Seconds())
}

// requestTimeoutCause turns a request that died on one of the client's own
// deadlines into its *timeoutError: the stall cause the watch recorded, else
// nil (classifyDeadline names the global one).
func requestTimeoutCause(ctx context.Context) error {
	var te *timeoutError
	if errors.As(context.Cause(ctx), &te) {
		return te
	}
	return nil
}
