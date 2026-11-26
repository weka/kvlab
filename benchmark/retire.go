package benchmark

import (
	"fmt"
	"sync/atomic"
	"time"
)

// A replay session instance whose prompt outgrows the context is retired: its
// remaining turns are not sent, since prompts only grow within an instance.
// Two things retire one, and they are counted apart because they mean
// different things: --limit-context is the operator's own cut on captured
// tokens, a server context-length 400 is the model's window.
const (
	retireLimitContext    = "limit_context"
	retireContextOverflow = "context_overflow"
)

// recordTypeInstanceRetired tags a retirement row in the request-data JSONL.
// It rides in the same stream as request rows and is told apart by
// record_type, like run_params, so the report can count retirements per arm.
const recordTypeInstanceRetired = "instance_retired"

// instanceRetiredRecord is one retirement, written when it happens.
type instanceRetiredRecord struct {
	RecordType string    `json:"record_type"`
	Time       time.Time `json:"ts"`
	SeriesGUID string    `json:"series_guid"`
	Turn       int       `json:"turn"`
	Reason     string    `json:"reason"`
	// CapturedInputTokens is the captured prompt of the request that
	// retired the instance: over --limit-context, or rejected by the server.
	CapturedInputTokens int    `json:"captured_input_tokens"`
	ErrorMessage        string `json:"error_message,omitempty"`
}

// retireCounters counts retirements by reason across concurrent sessions.
type retireCounters struct {
	limitContext    atomic.Int64
	contextOverflow atomic.Int64
}

func (c *retireCounters) add(reason string) {
	switch reason {
	case retireLimitContext:
		c.limitContext.Add(1)
	case retireContextOverflow:
		c.contextOverflow.Add(1)
	}
}

func (c *retireCounters) load() retireTotals {
	return retireTotals{LimitContext: c.limitContext.Load(), ContextOverflow: c.contextOverflow.Load()}
}

// retireTotals is one snapshot of retirements by reason.
type retireTotals struct {
	LimitContext    int64
	ContextOverflow int64
}

func (r retireTotals) total() int64 { return r.LimitContext + r.ContextOverflow }

// summary is the summary's line body.
func (r retireTotals) summary() string {
	return fmt.Sprintf("%d session instances: %d over --limit-context, %d by server context-length 400",
		r.total(), r.LimitContext, r.ContextOverflow)
}

// progress is the progress line's segment, "" when nothing was retired.
func (r retireTotals) progress() string {
	if r.total() == 0 {
		return ""
	}
	return fmt.Sprintf(" retired=limit:%d,overflow:%d", r.LimitContext, r.ContextOverflow)
}

// retiredByReason counts an arm's retirement rows by reason, for the report.
func retiredByReason(rows []instanceRetiredRecord) map[string]int {
	if len(rows) == 0 {
		return nil
	}
	m := map[string]int{}
	for _, r := range rows {
		m[r.Reason]++
	}
	return m
}
