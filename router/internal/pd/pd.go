// Package pd is vanilla vLLM prefill/decode disaggregation over NIXL.
//
// A static subset of a pool's backends are prefill (P) nodes and the rest are
// decode (D) nodes. Every request goes to a P node first, as a copy limited to
// one token that asks the engine to leave its KV for a remote decoder; the
// engine answers with kv_transfer_params naming where it is. The request itself
// then goes to a D node carrying those params, which pulls the blocks it does
// not already hold and streams the answer.
//
// With Dynamic set the P set is not static: it is the prefill manager's
// nodes out of decode mode. The configured prefill URLs are its floor, and the
// manager drains decode nodes into it and releases them back as the prefill
// load demands, as for the prefill split (see prefill/pdmode.go).
package pd

import (
	"encoding/json"
	"strings"

	"github.com/weka/kvlab/router/internal/metrics"
	"github.com/weka/kvlab/router/internal/prefill"
	"github.com/weka/kvlab/router/internal/registry"
)

// Set is the P/D split of one pool.
type Set struct {
	prefill map[string]struct{}
	M       *metrics.PDMetrics
	// Dynamic, when set, owns the P set: the URLs above are only its floor,
	// and every prefill leg is a job of the manager, charged against its
	// budget.
	Dynamic *prefill.Manager
}

// New builds a set from the prefill backends' URLs.
func New(pool string, prefillURLs []string) *Set {
	s := &Set{prefill: map[string]struct{}{}, M: metrics.ForPD(pool)}
	for _, u := range prefillURLs {
		s.prefill[Normalize(u)] = struct{}{}
	}
	return s
}

// Normalize makes backend URLs comparable.
func Normalize(u string) string { return strings.TrimRight(strings.TrimSpace(u), "/") }

// IsPrefill reports whether b is a P node.
func (s *Set) IsPrefill(b *registry.Backend) bool {
	if s.Dynamic != nil {
		return s.Dynamic.IsPrefill(b)
	}
	_, ok := s.prefill[Normalize(b.URL)]
	return ok
}

// Role is b's role, for display: "prefill" or "decode", or under Dynamic also
// "draining".
func (s *Set) Role(b *registry.Backend) string {
	if s.Dynamic != nil {
		if m := s.Dynamic.ModeOf(b); m != prefill.Decode {
			return m.String()
		}
		if s.Dynamic.IsPrefill(b) {
			return prefill.Prefill.String() // a floor node not met yet
		}
		return prefill.Decode.String()
	}
	if s.IsPrefill(b) {
		return "prefill"
	}
	return "decode"
}

// PrefillNodes keeps the P nodes of cands.
func (s *Set) PrefillNodes(cands []*registry.Backend) []*registry.Backend {
	if s.Dynamic != nil {
		return s.Dynamic.PrefillNodes(cands)
	}
	return s.filter(cands, true)
}

// DecodeNodes keeps the D nodes of cands. A node in the P set, draining
// included, is never one.
func (s *Set) DecodeNodes(cands []*registry.Backend) []*registry.Backend {
	if s.Dynamic != nil {
		return s.Dynamic.DecodeNodes(cands)
	}
	return s.filter(cands, false)
}

func (s *Set) filter(cands []*registry.Backend, want bool) []*registry.Backend {
	out := make([]*registry.Backend, 0, len(cands))
	for _, b := range cands {
		if s.IsPrefill(b) == want {
			out = append(out, b)
		}
	}
	return out
}

// Params is the kv_transfer_params object of a request or response, verbatim.
type Params = json.RawMessage

var prefillParams = json.RawMessage(`{"do_remote_decode":true,"do_remote_prefill":false,` +
	`"remote_engine_id":null,"remote_block_ids":null,"remote_host":null,"remote_port":null}`)

// PrefillBody turns a request into its prefill leg: one token, not streamed,
// asking to leave its KV for a remote decoder. A body that is not a JSON object
// is returned unchanged.
func PrefillBody(body []byte) []byte {
	doc, ok := parse(body)
	if !ok {
		return body
	}
	doc["max_tokens"] = json.RawMessage("1")
	if _, has := doc["max_completion_tokens"]; has {
		doc["max_completion_tokens"] = json.RawMessage("1")
	}
	doc["stream"] = json.RawMessage("false")
	delete(doc, "stream_options")
	doc["kv_transfer_params"] = prefillParams
	return encode(doc, body)
}

// DecodeBody sets the request's kv_transfer_params to the prefill leg's answer.
func DecodeBody(body []byte, params Params) []byte {
	doc, ok := parse(body)
	if !ok {
		return body
	}
	doc["kv_transfer_params"] = params
	return encode(doc, body)
}

// ParseParams extracts the top-level kv_transfer_params object from a prefill
// leg's response. ok is false when it is absent, null or not an object.
func ParseParams(resp []byte) (Params, bool) {
	var doc struct {
		P json.RawMessage `json:"kv_transfer_params"`
	}
	if json.Unmarshal(resp, &doc) != nil {
		return nil, false
	}
	p := strings.TrimSpace(string(doc.P))
	if !strings.HasPrefix(p, "{") {
		return nil, false
	}
	return doc.P, true
}

func parse(body []byte) (map[string]json.RawMessage, bool) {
	var m map[string]json.RawMessage
	if json.Unmarshal(body, &m) != nil || m == nil {
		return nil, false
	}
	return m, true
}

func encode(doc map[string]json.RawMessage, orig []byte) []byte {
	out, err := json.Marshal(doc)
	if err != nil {
		return orig
	}
	return out
}

// URLs is the configured prefill URLs, normalised and in no particular order.
func (s *Set) URLs() []string {
	out := make([]string, 0, len(s.prefill))
	for u := range s.prefill {
		out = append(out, u)
	}
	return out
}

// Event is a change in whether a named prefill URL is present in the pool.
type Event struct {
	URL  string
	Kind EventKind
}

// EventKind says what happened to a named prefill URL.
type EventKind string

const (
	// Missing: never seen, and the grace period for discovery to find it is over.
	Missing EventKind = "missing"
	// Gone: was present, now is not.
	Gone EventKind = "gone"
	// Back: was reported Missing or Gone and is present again.
	Back EventKind = "back"
)

// Watch reports, over successive Check calls, which named prefill URLs are
// absent from a pool whose backends are discovered rather than listed. Roles
// are assigned by URL, so an absent P node is not an error: the request just
// falls back to decode. It is only worth telling the operator about.
type Watch struct {
	set      *Set
	seen     map[string]bool // present at the last check
	reported map[string]bool // a Missing or Gone is outstanding
}

// NewWatch watches the set's URLs.
func NewWatch(s *Set) *Watch {
	return &Watch{set: s, seen: map[string]bool{}, reported: map[string]bool{}}
}

// Check compares the pool's backends with the named prefill URLs and returns
// what changed since the last call. A URL not yet seen is reported Missing only
// once graceOver is true, so discovery has time to sync first.
func (w *Watch) Check(backends []*registry.Backend, graceOver bool) []Event {
	present := map[string]bool{}
	for _, b := range backends {
		present[Normalize(b.URL)] = true
	}
	var out []Event
	for u := range w.set.prefill {
		was, have := w.seen[u], present[u]
		switch {
		case have && w.reported[u]:
			w.reported[u] = false
			out = append(out, Event{u, Back})
		case !have && was:
			w.reported[u] = true
			out = append(out, Event{u, Gone})
		case !have && !was && graceOver && !w.reported[u]:
			w.reported[u] = true
			out = append(out, Event{u, Missing})
		}
		w.seen[u] = have
	}
	return out
}
