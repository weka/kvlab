package gateway

import (
	"context"
	"hash/fnv"
	"regexp"
	"sync"

	"github.com/weka/kvlab/kvcache"
	"github.com/weka/kvlab/router/internal/obs"
)

// pdLegDiag logs, for a sample of conversations, how each of their prefill
// legs was charged and how far its prompt matches the conversation's previous
// leg, unit by unit, as the router's prefix tree sees it. A charge far above
// what the P node computes means the tree's record of the prompt broke off
// early; this line says where, without logging every leg.
//
// A conversation is known by its request id: kvlab's replay sends
// "<series>-t<turn>-<suffix>", and the part before "-t<turn>" names the
// conversation. Requests with ids of another shape are not sampled. One
// conversation in pdDiagSampleEvery is followed, chosen by a hash of its name,
// so every leg of a followed conversation is logged and the rest never are.
type pdLegDiag struct {
	mu   sync.Mutex
	prev map[string][]uint64
}

const (
	pdDiagSampleEvery      = 64
	pdDiagMaxConversations = 4096
)

var turnRequestID = regexp.MustCompile(`^(.+)-t\d+-[^-]*$`)

// pdDiagConversation is the followed conversation r's request id belongs to.
func pdDiagConversation(id string) (string, bool) {
	m := turnRequestID.FindStringSubmatch(id)
	if m == nil {
		return "", false
	}
	h := fnv.New32a()
	_, _ = h.Write([]byte(m[1]))
	return m[1], h.Sum32()%pdDiagSampleEvery == 0
}

func (d *pdLegDiag) observe(ctx context.Context, units []kvcache.Unit, c pdCharge) {
	conv, ok := pdDiagConversation(obs.RequestID(ctx))
	if !ok {
		return
	}
	hashes := make([]uint64, len(units))
	for i, u := range units {
		hashes[i] = u.Hash
	}
	d.mu.Lock()
	if d.prev == nil {
		d.prev = map[string][]uint64{}
	}
	prev, seen := d.prev[conv]
	if !seen && len(d.prev) >= pdDiagMaxConversations {
		for k := range d.prev {
			delete(d.prev, k)
			break
		}
	}
	d.prev[conv] = hashes
	d.mu.Unlock()

	// diverge_unit is the first unit of this prompt that differs from the
	// previous leg's (its length when one extends the other); -1 on a
	// conversation's first sampled leg.
	diverge, shared := -1, int64(-1)
	if seen {
		diverge, shared = 0, 0
		for diverge < len(prev) && diverge < len(hashes) && prev[diverge] == hashes[diverge] {
			shared += int64(units[diverge].Tokens)
			diverge++
		}
	}
	obs.Logger(ctx).Info("pd leg charge", "conversation", conv,
		"units", len(units), "prev_units", len(prev), "diverge_unit", diverge, "shared_with_prev_tokens", shared,
		"total_tokens", c.total, "prefill_tree_tokens", c.local, "decode_tree_tokens", c.decode,
		"stored_tokens", c.stored, "charge_tokens", c.charge)
}
