package serve

import (
	"fmt"
	"log/slog"
	"net/http"

	"github.com/weka/kvlab/router/internal/anthropicapi"
)

// AnthropicOptions configures the Anthropic Messages API translation mode
// (`--anthropic-api`). With it on, POST /v1/messages and
// /v1/messages/count_tokens are answered by translating to the fleet's
// /v1/chat/completions and back, so an Anthropic client — Claude Code among
// them — can use an OpenAI-only backend, tools and streaming included.
type AnthropicOptions struct {
	// ModelMap entries are "<pattern>=<model>": comma-separated substrings of
	// the client's model name (or "*"), first match wins. An unmatched name
	// is forwarded as sent.
	ModelMap []string
	// ThinkingKwargs are chat_template_kwargs keys set to the request's
	// thinking on/off. Empty sends none.
	ThinkingKwargs []string
	// Thinking is "request" (default), "always" or "never": when backend
	// reasoning is returned as thinking blocks.
	Thinking string
	// MaxTokensCap clamps the client's max_tokens; 0 leaves it.
	MaxTokensCap int
	// DebugDir records every exchange in both formats; empty is off.
	DebugDir string
}

func (a *AnthropicOptions) middleware(opts Options, next http.Handler, log *slog.Logger) (http.Handler, error) {
	cfg := anthropicapi.Config{
		ThinkingKwargs: a.ThinkingKwargs,
		MaxTokensCap:   a.MaxTokensCap,
		MaxBodyBytes:   opts.MaxBodyBytes,
		UserPrefix:     opts.UserPrefix,
		DebugDir:       a.DebugDir,
		Log:            log,
	}
	switch a.Thinking {
	case "", "request":
		cfg.Thinking = anthropicapi.ThinkingRequest
	case "always":
		cfg.Thinking = anthropicapi.ThinkingAlways
	case "never":
		cfg.Thinking = anthropicapi.ThinkingNever
	default:
		return nil, fmt.Errorf("--anthropic-thinking: want request, always or never, got %q", a.Thinking)
	}
	for _, m := range a.ModelMap {
		r, err := anthropicapi.ParseModelRule(m)
		if err != nil {
			return nil, fmt.Errorf("--anthropic-model-map: %w", err)
		}
		cfg.ModelMap = append(cfg.ModelMap, r)
	}
	log.Info("serving the Anthropic Messages API by translation to chat completions",
		"model_map", a.ModelMap, "thinking", cfg.Thinking, "thinking_kwargs", a.ThinkingKwargs,
		"max_tokens_cap", a.MaxTokensCap)
	if a.DebugDir != "" {
		log.Warn("--anthropic-debug-dir records full request and response bodies", "dir", a.DebugDir)
	}
	return anthropicapi.Middleware(cfg, next), nil
}
