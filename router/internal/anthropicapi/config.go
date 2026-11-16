package anthropicapi

import (
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/weka/kvlab/router/internal/clock"
)

// ThinkingMode decides when backend reasoning is shown as thinking blocks.
type ThinkingMode string

const (
	// ThinkingRequest surfaces reasoning only when the request enabled
	// thinking, which is what Anthropic itself does.
	ThinkingRequest ThinkingMode = "request"
	// ThinkingAlways surfaces reasoning whenever the backend produces it.
	ThinkingAlways ThinkingMode = "always"
	// ThinkingNever drops it.
	ThinkingNever ThinkingMode = "never"
)

// ModelRule maps client model names to a backend model. Patterns are
// comma-separated case-insensitive substrings, or "*" for every name — the
// same syntax as a route.
type ModelRule struct {
	Patterns []string
	Model    string
}

// ParseModelRule reads "<pattern>=<model>".
func ParseModelRule(s string) (ModelRule, error) {
	i := strings.LastIndexByte(s, '=')
	if i <= 0 || i == len(s)-1 {
		return ModelRule{}, fmt.Errorf("model map %q: want <pattern>=<model>", s)
	}
	pat := strings.TrimSpace(s[:i])
	model := strings.TrimSpace(s[i+1:])
	var pats []string
	if pat != "*" {
		for _, p := range strings.Split(pat, ",") {
			if p = strings.ToLower(strings.TrimSpace(p)); p != "" {
				pats = append(pats, p)
			}
		}
		if len(pats) == 0 {
			return ModelRule{}, fmt.Errorf("model map %q: empty pattern", s)
		}
	}
	return ModelRule{Patterns: pats, Model: model}, nil
}

func (r ModelRule) matches(model string) bool {
	if len(r.Patterns) == 0 {
		return true
	}
	lower := strings.ToLower(model)
	for _, p := range r.Patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// Config is the translation mode's configuration.
type Config struct {
	// ModelMap rewrites the client's model name before routing; first match
	// wins, and an unmatched name is forwarded unchanged (a route's own
	// `as <model>` or auto-model may still rewrite it).
	ModelMap []ModelRule
	// ThinkingKwargs are the chat_template_kwargs keys set to the request's
	// thinking on/off. Empty never sends any.
	ThinkingKwargs []string
	Thinking       ThinkingMode
	// MaxTokensCap clamps max_tokens. Claude Code asks for tens of thousands of
	// output tokens on every request, which a backend with a smaller context
	// rejects outright. 0 forwards the client's value.
	MaxTokensCap int
	// MaxBodyBytes bounds the inbound body, as the gateway would.
	MaxBodyBytes int64
	// UserPrefix also serves /<user>/v1/messages, mirroring the gateway's
	// --user-prefix.
	UserPrefix bool
	// PingInterval is how long a stream may emit nothing before a ping is
	// sent, which keeps an idle-timeout client attached while the backend
	// reasons silently. 0 uses the default.
	PingInterval time.Duration
	// DebugDir, when set, appends every exchange in both wire formats to
	// DebugDir/anthropic-exchanges.jsonl. Bodies only; never headers.
	DebugDir string

	Clock clock.Clock
	Log   *slog.Logger
}

const defaultPingInterval = 10 * time.Second

func (c *Config) mapModel(model string) string {
	for _, r := range c.ModelMap {
		if r.matches(model) {
			return r.Model
		}
	}
	return model
}

func (c *Config) withDefaults() {
	if c.Clock == nil {
		c.Clock = clock.Real{}
	}
	if c.Log == nil {
		c.Log = slog.Default()
	}
	if c.PingInterval <= 0 {
		c.PingInterval = defaultPingInterval
	}
	if c.Thinking == "" {
		c.Thinking = ThinkingRequest
	}
	if c.MaxBodyBytes <= 0 {
		c.MaxBodyBytes = 64 << 20
	}
}
