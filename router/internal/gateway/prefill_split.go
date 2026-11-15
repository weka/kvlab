package gateway

import (
	"net/http"
	"strconv"
	"time"

	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/prefill"
	"github.com/weka/kvlab/router/internal/proxy"
	"github.com/weka/kvlab/router/internal/registry"
)

// splitAndServe answers a request its backend refused as a large prefill,
// without the caller ever seeing the refusal.
//
// The prefill runs on a node the manager has set aside, as a copy of the
// request limited to one token. The request itself then goes through ordinary
// decode routing, over decode nodes only, and its response is the caller's.
// Both carry the bypass, so neither can be refused again: if the shared KV tier
// lagged the job, the decode node recomputes the rest instead of looping.
//
// When no prefill node can be used the request goes back to the backend that
// refused it, with the bypass, and is served there.
func (s *Server) splitAndServe(
	w http.ResponseWriter, r *http.Request, target Target,
	rr *policy.RoutingRequest, body []byte, rejected proxy.Result,
	accepted proxy.OnAccepted, outcome proxy.OnOutcome, auth proxy.Auth,
) proxy.Result {
	m := target.Split
	m.CountRejection()
	// Everything sent from here carries the bypass, so a refusal of it is not
	// absorbed: it goes to the caller instead of starting another round.
	r = r.WithContext(proxy.WithoutRejectCapture(r.Context()))
	bypassed := prefill.WithBypass(body)
	// A fresh decision: the rejected attempt's selection state is not this
	// request's final one.
	rr.PolicyState = nil

	job, err := m.Reserve(s.healthy(target), rejected.PrefillTokens, s.prefillPicker(target, m, rr, r))
	if err != nil {
		m.FallbackNoNode()
		return s.serveOnOriginal(w, r, target, rr, bypassed, rejected.Backend, accepted, outcome, auth)
	}
	defer job.Done()

	if err := m.WaitReady(r.Context(), job); err != nil {
		return proxy.Result{Err: err}
	}
	ok := s.runPrefillJob(r, m, target, job.Backend, rr, prefill.JobBody(body), auth)
	m.CountJob(ok)
	if !ok {
		m.FallbackJobFailed()
	} else if c, isC := target.SplitSelector.(policy.Committer); isC {
		// The node now holds the prefix: remember it in the prefill flow, so a
		// later job of the same conversation prefers it. Never in the decode
		// flow, where the node is not a candidate and a mark would make the
		// request's real holders look unavailable to the resend.
		leg := *rr
		leg.PolicyState = nil
		c.Commit(job.Backend, &leg)
		// The prompt's KV, up to the tier's storage cut, can now be loaded by
		// any decode node: the resend's cost counts it as held everywhere
		// without marking it in the decode tree.
		var total int64
		for _, u := range rr.Units {
			total += int64(u.Tokens)
		}
		rr.PreloadedTokens = m.StoreCut(total)
	}
	job.Done()

	rr.PolicyState = nil
	return s.serveWithCapacityRetry(w, r, target, rr, bypassed, accepted, outcome, auth)
}

// prefillPicker chooses among eligible prefill nodes with the pool's own flow,
// so the choice sees the same prefix knowledge and load rules as ordinary
// routing. It selects on a copy of the request: the flow's per-request state
// belongs to the decode decision that comes after.
//
// The job is costed as a prefill-only request (see RoutingRequest.PrefillOnly)
// whose queue on each node is the manager's pending job tokens there. The prefix
// an earlier job of the prefill flow stored, up to the tier's storage cut, is
// loadable by every node, so it is credited to all of them as preloaded: a node
// that holds it locally keeps an advantage only for the tokens past it.
func (s *Server) prefillPicker(t Target, m *prefill.Manager, rr *policy.RoutingRequest, r *http.Request) prefill.Picker {
	var stored int64
	if h, ok := t.SplitSelector.(heldDepther); ok {
		stored = m.StoreCut(h.HeldDepth(rr))
	}
	return func(eligible []*registry.Backend, queued func(*registry.Backend) int64) (*registry.Backend, error) {
		sel := *rr
		sel.PolicyState = nil
		if t.SplitSelector != nil {
			sel.PrefillOnly, sel.PrefillQueue, sel.PreloadedTokens = true, queued, stored
			return t.SplitSelector.Select(r.Context(), eligible, &sel)
		}
		return t.Selector.Select(r.Context(), eligible, &sel)
	}
}

// heldDepther is a flow that can say how deep a prefix any of its backends is
// recorded holding.
type heldDepther interface {
	HeldDepth(rr *policy.RoutingRequest) int64
}

// runPrefillJob sends the job to its node through the ordinary proxy path, so
// the lease, circuit breaker, credentials and timeouts all apply, and discards
// the response. It does not commit the prefix to the flow: the node is outside
// decode routing, and a mark there would make the request's real holders look
// unavailable to the decode decision that follows.
//
// A 429 means the node's own admission cap is below the router's view, or a
// race lost: the job holds its slot and tries again after a short backoff, the
// node's Retry-After when it sent one, until it is served or the caller leaves.
// Only another failure ends the job unsuccessfully.
func (s *Server) runPrefillJob(r *http.Request, m *prefill.Manager, target Target, b *registry.Backend,
	rr *policy.RoutingRequest, body []byte, auth proxy.Auth) bool {
	job := *rr
	job.PolicyState = nil
	// The prefill flow, when there is one, learns of a node's refusals and
	// acceptances like the fixed P/D flow does, so a refusing node is passed
	// over for its TTL.
	var outcome proxy.OnOutcome
	if o, ok := target.SplitSelector.(policy.Observer); ok {
		outcome = func(b *registry.Backend, status int) {
			if status == http.StatusTooManyRequests {
				o.OnRefused(b)
			} else if status >= 200 && status < 500 {
				o.OnAccepted(b)
			}
		}
	}
	for attempt := 0; ; attempt++ {
		dw := &discardWriter{h: http.Header{}}
		res := s.px.Serve(dw, r, []*registry.Backend{b}, policy.LeastOutstanding{},
			s.dialect, &job, body, nil, outcome, auth)
		if res.Err == nil && !res.PrefillRejected && res.Status >= 200 && res.Status < 300 {
			return true
		}
		if res.Status != http.StatusTooManyRequests || res.Err != nil {
			return false
		}
		m.CountJobRetry()
		wait := backoffAt(attempt, time.Hour)
		if ra, err := strconv.Atoi(dw.h.Get("Retry-After")); err == nil && ra >= 0 {
			wait = min(time.Duration(ra)*time.Second, maxJobRetryWait)
		}
		select {
		case <-s.cfg.Clock.After(wait):
		case <-r.Context().Done():
			return false
		}
	}
}

// maxJobRetryWait bounds a node's Retry-After, so one odd header cannot park a
// request that is holding a slot.
const maxJobRetryWait = 5 * time.Second

// serveOnOriginal resends a refused request to the backend that refused it. A
// backend that has since left decode mode or health is not sent decode work;
// the request is routed normally instead.
func (s *Server) serveOnOriginal(
	w http.ResponseWriter, r *http.Request, target Target,
	rr *policy.RoutingRequest, body []byte, orig *registry.Backend,
	accepted proxy.OnAccepted, outcome proxy.OnOutcome, auth proxy.Auth,
) proxy.Result {
	if orig != nil && orig.Available() && target.Split.ModeOf(orig) == prefill.Decode {
		return s.px.Serve(w, r, []*registry.Backend{orig}, policy.LeastOutstanding{},
			s.dialect, rr, body, accepted, outcome, auth)
	}
	return s.serveWithCapacityRetry(w, r, target, rr, body, accepted, outcome, auth)
}

// discardWriter swallows a prefill job's response.
type discardWriter struct{ h http.Header }

func (d *discardWriter) Header() http.Header         { return d.h }
func (d *discardWriter) Write(p []byte) (int, error) { return len(p), nil }
func (d *discardWriter) WriteHeader(int)             {}
