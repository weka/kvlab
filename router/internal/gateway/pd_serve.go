package gateway

import (
	"bytes"
	"net/http"
	"strconv"
	"time"

	"github.com/weka/kvlab/router/internal/pd"
	"github.com/weka/kvlab/router/internal/policy"
	"github.com/weka/kvlab/router/internal/prefill"
	"github.com/weka/kvlab/router/internal/proxy"
	"github.com/weka/kvlab/router/internal/registry"
)

// pdRoute reports whether a route class carries NIXL kv_transfer_params. Only
// the OpenAI generation endpoints do: the Anthropic messages surface has no
// field for them, so it is served over the D nodes alone.
func pdRoute(class string) bool { return class == "chat" || class == "completions" }

// pdServe serves a request as a prefill leg on a P node followed by a decode
// leg on a D node, as vLLM's own disaggregation proxy does.
//
// Both legs are routed by the pool's own flow, restricted to their node set, so
// each sees the prefix knowledge and load rules of ordinary routing, and each
// records the prefix against the node that served it. A prefill leg that cannot
// be completed sends the request to a D node without kv_transfer_params, which
// recomputes the prompt.
func (s *Server) pdServe(
	w http.ResponseWriter, r *http.Request, target Target,
	rr *policy.RoutingRequest, body []byte,
	accepted proxy.OnAccepted, outcome proxy.OnOutcome, auth proxy.Auth,
) proxy.Result {
	set := target.PD
	decodeBody := body
	if params, ok := s.runPDPrefill(r, target, rr, body, outcome, auth); ok {
		decodeBody = pd.DecodeBody(body, params)
		rr.KVTransfer = true
	}
	rr.PolicyState = nil
	start := s.cfg.Clock.Now()
	res := s.serveWithCapacityRetry(w, r, target, rr, decodeBody, accepted, outcome, auth)
	set.M.DecodeSeconds.Observe(s.cfg.Clock.Now().Sub(start).Seconds())
	if !rr.KVTransfer && res.Err == nil && res.Status >= 200 && res.Status < 300 {
		// Without a prefill leg the decode node computed the prompt, and
		// stored it.
		set.Stored.Record(rr)
	}
	return res
}

// runPDPrefill runs the prefill leg and returns the kv_transfer_params the P
// node answered with. ok is false when the request must fall back to a D node
// alone.
//
// A router-side saturation of every P node, or a 429 from the chosen one, is
// waited out and retried, like a prefill-split job: the leg holds its place
// until it is served, --retry-time-limit runs out when that is set, or the
// caller leaves.
//
// Under dynamic P/D the leg is a job of the prefill split's manager instead:
// charged against a P node's budget, it waits in the router while every P node
// is full (which is what drains another decode node into prefill), then goes to
// the node the manager placed it on, once that node has finished draining. It
// is counted as a prefill job as well as a leg.
func (s *Server) runPDPrefill(
	r *http.Request, target Target, rr *policy.RoutingRequest, body []byte,
	outcome proxy.OnOutcome, auth proxy.Auth,
) (pd.Params, bool) {
	set := target.PD
	m := set.M
	start := s.cfg.Clock.Now()
	defer func() { m.PrefillSeconds.Observe(s.cfg.Clock.Now().Sub(start).Seconds()) }()

	leg := *rr
	leg.PrefillOnly = true
	sel := target.PrefillSelector
	var accepted proxy.OnAccepted
	if c, ok := sel.(policy.Committer); ok {
		accepted = func(b *registry.Backend) { c.Commit(b, &leg) }
	}
	if o, ok := sel.(policy.Observer); ok {
		outcome = func(b *registry.Backend, status int) {
			if status == http.StatusTooManyRequests {
				o.OnRefused(b)
			} else if status >= 200 && status < 500 {
				o.OnAccepted(b)
			}
		}
	}
	pbody := pd.PrefillBody(body)
	deadline := start.Add(s.cfg.RetryTimeLimit)

	cands := func() []*registry.Backend { return set.PrefillNodes(s.healthy(target)) }
	mgr := set.Dynamic
	if mgr != nil {
		c := s.pdLegTokens(target, mgr, rr)
		m.HeldPrefill.Observe(float64(c.local))
		m.HeldDecode.Observe(float64(c.decode))
		m.HeldStored.Observe(float64(c.record))
		s.pdDiag.observe(r.Context(), rr.Units, c)
		// The manager packs legs onto its prefill nodes in a fixed order; no
		// policy is consulted.
		job, err := mgr.Reserve(s.healthy(target), c.charge, nil)
		if err != nil {
			m.FallbackNoNode.Inc()
			mgr.FallbackNoNode()
			return nil, false
		}
		defer job.Done()
		if err := mgr.WaitReady(r.Context(), job); err != nil {
			mgr.CountJobCancelled("abandoned")
			return nil, false // the caller left; its decode leg ends with it
		}
		// The manager chose the node and bounds its legs (budget, and
		// --max-node-concurrency when set): the leg goes there alone.
		only := []*registry.Backend{job.Backend}
		cands, sel = func() []*registry.Backend { return only }, policy.LeastOutstanding{}
	}
	jobFailed := func(reason string, cancelled bool) {
		switch {
		case mgr == nil:
		case cancelled:
			mgr.CountJobCancelled(reason)
		default:
			mgr.CountJobFailed(reason)
			mgr.FallbackJobFailed()
		}
	}

	dispatched := s.cfg.Clock.Now()
	for attempt := 0; ; attempt++ {
		cands := cands()
		if len(cands) == 0 {
			m.FallbackNoNode.Inc()
			return nil, false
		}
		leg.PolicyState = nil
		cw := &captureWriter{h: http.Header{}}
		res := s.px.Serve(cw, r, cands, sel, s.dialect, &leg, pbody, accepted, outcome, auth)

		refused := isCapacityRefusal(res.Err) || (res.Err == nil && res.Status == http.StatusTooManyRequests)
		if refused && r.Context().Err() == nil &&
			(s.cfg.RetryTimeLimit <= 0 || s.cfg.Clock.Now().Before(deadline)) {
			m.Retries.Inc()
			wait := backoffAt(attempt, time.Hour)
			if ra, err := strconv.Atoi(cw.h.Get("Retry-After")); err == nil && ra >= 0 {
				wait = min(time.Duration(ra)*time.Second, maxJobRetryWait)
			}
			select {
			case <-s.cfg.Clock.After(wait):
				continue
			case <-r.Context().Done():
			}
		}

		if res.Err != nil || res.Status < 200 || res.Status >= 300 {
			m.LegFailed.Inc()
			m.FallbackFailed.Inc()
			f := classifyJob(r, res)
			jobFailed(f.reason, f.cancelled)
			return nil, false
		}
		params, ok := pd.ParseParams(cw.buf.Bytes())
		if !ok {
			m.LegNoParams.Inc()
			m.FallbackNoParams.Inc()
			jobFailed("other", false)
			return nil, false
		}
		m.LegOK.Inc()
		m.ServiceSeconds.Observe(s.cfg.Clock.Now().Sub(dispatched).Seconds())
		// The P node stored the prompt to the tier as it computed it.
		set.Stored.Record(rr)
		if mgr != nil {
			mgr.CountJob(true)
		}
		return params, true
	}
}

// pdLegTokens is what a prefill leg is charged against a P node's budget
// under dynamic P/D, and the prefix the shared KV tier is credited with.
//
// Every earlier leg stored its prompt to the tier, and the tier serves it to
// any node, so a P node's work on a prompt the fleet has prefilled before is
// a load, not a recompute. The router knows such a prompt by either flow's
// prefix tree: the P tree records each leg's prompt on its P node, the D tree
// the same prompt on the decode node that served it, and the D tree keeps it
// when the P node's marks are gone (its tail TTL, or a release back to
// decode). The deeper of the two, cut to --prefill-store-unit-tokens, is
// credited as stored, as for the reject mode's jobs; the P tree's own depth is
// a local hit and is credited whole.
func (s *Server) pdLegTokens(t Target, m *prefill.Manager, rr *policy.RoutingRequest) pdCharge {
	var c pdCharge
	for _, u := range rr.Units {
		c.total += int64(u.Tokens)
	}
	if h, ok := t.PrefillSelector.(heldDepther); ok {
		c.local = h.HeldDepth(rr)
	}
	if h, ok := t.Selector.(heldDepther); ok {
		c.decode = h.HeldDepth(rr)
	}
	c.record = t.PD.Stored.Depth(rr)
	c.stored = m.StoreCut(max(c.local, c.decode, c.record))
	c.charge = pdLegCharge(c.total, c.local, c.stored, m.LoadWeight())
	return c
}

// pdCharge is how a leg's budget charge was made up, in router-estimated
// tokens: the prompt, the deepest prefix the P tree and the D tree record,
// what is credited as stored in the tier, and the charge.
type pdCharge struct {
	total, local, decode, record, stored, charge int64
}

// pdLegCharge is a leg's budget charge: the tokens it computes, the prompt
// less what a P node already holds or can load from the tier (at least one),
// plus loadWeight times the tokens it loads from the tier. A P node that is
// slow because of its loads rather than its compute shows only through the
// load term; at 0 the charge is compute alone.
func pdLegCharge(total, local, stored int64, loadWeight float64) int64 {
	compute := max(total-max(local, stored), 1)
	loaded := max(stored-local, 0)
	return compute + int64(loadWeight*float64(loaded))
}

// captureWriter keeps a prefill leg's response, which is small: one token and
// the transfer params.
type captureWriter struct {
	h   http.Header
	buf bytes.Buffer
}

func (c *captureWriter) Header() http.Header         { return c.h }
func (c *captureWriter) Write(p []byte) (int, error) { return c.buf.Write(p) }
func (c *captureWriter) WriteHeader(int)             {}
