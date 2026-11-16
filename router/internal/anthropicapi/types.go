// Package anthropicapi serves the Anthropic Messages API in front of an
// OpenAI-only fleet by translating each exchange to Chat Completions and back.
//
// It sits OUTSIDE the gateway: a /v1/messages request is rewritten into a
// /v1/chat/completions request and handed to the ordinary routing handler, so
// it is authorised, routed with prefix affinity, retried and measured exactly
// like native OpenAI traffic. The response — a JSON body or an SSE stream — is
// translated back as it is written. Nothing below the gateway knows a
// translation happened.
//
// The translation is deliberately lossy in one direction only: Anthropic
// features with no Chat Completions counterpart (server tools, prompt caching
// directives, citations, thinking signatures) are dropped on the way in rather
// than refused, because a client such as Claude Code sends them on every
// request and refusing would make the mode useless.
package anthropicapi

import (
	"bytes"
	"encoding/json"
	"errors"
)

// MessagesRequest is the subset of POST /v1/messages the translation reads.
// Unknown fields are ignored, which is what keeps a newer client working.
type MessagesRequest struct {
	Model         string          `json:"model"`
	MaxTokens     *int            `json:"max_tokens"`
	System        json.RawMessage `json:"system"`
	Messages      []Message       `json:"messages"`
	Tools         []Tool          `json:"tools"`
	ToolChoice    *ToolChoice     `json:"tool_choice"`
	Temperature   *float64        `json:"temperature"`
	TopP          *float64        `json:"top_p"`
	TopK          *int            `json:"top_k"`
	StopSequences []string        `json:"stop_sequences"`
	Stream        bool            `json:"stream"`
	Metadata      *Metadata       `json:"metadata"`
	Thinking      *Thinking       `json:"thinking"`
}

// Metadata carries the caller's opaque end-user id.
type Metadata struct {
	UserID string `json:"user_id"`
}

// Thinking is the extended-thinking request.
type Thinking struct {
	Type         string `json:"type"` // enabled | disabled | adaptive
	BudgetTokens int    `json:"budget_tokens"`
}

// enabled reports whether the caller asked for thinking at all.
func (t *Thinking) enabled() bool {
	return t != nil && (t.Type == "enabled" || t.Type == "adaptive")
}

// Message is one conversation turn. Content is a string or an array of blocks.
type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// Block is any content block; which fields are meaningful depends on Type.
type Block struct {
	Type string `json:"type"`

	// text
	Text string `json:"text,omitempty"`

	// image, document
	// Source is decoded lazily (see source): an image's or document's is an
	// object, but other block types use the same key for a plain string, and
	// a block this translation does not read must never fail the request.
	Source json.RawMessage `json:"source,omitempty"`
	Title  string          `json:"title,omitempty"`

	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`

	// thinking, redacted_thinking
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// source decodes an image's or document's payload.
func (b Block) source() *Source {
	var s Source
	if len(b.Source) == 0 || json.Unmarshal(b.Source, &s) != nil {
		return nil
	}
	return &s
}

// Source is an image or document payload.
type Source struct {
	Type      string          `json:"type"` // base64 | url | text | content
	MediaType string          `json:"media_type,omitempty"`
	Data      string          `json:"data,omitempty"`
	URL       string          `json:"url,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
}

// Tool is a tool definition. A client tool has an input_schema; a server tool
// (web search, code execution, ...) has a versioned type and none.
type Tool struct {
	Type        string          `json:"type,omitempty"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

// isClientTool reports whether the backend can be asked to call this tool.
// Server tools run inside Anthropic's API; an OpenAI backend cannot execute
// them and has no way to return their results.
func (t Tool) isClientTool() bool {
	return t.Type == "" || t.Type == "custom" || len(t.InputSchema) > 0
}

// ToolChoice constrains tool use.
type ToolChoice struct {
	Type                   string `json:"type"` // auto | any | tool | none
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}

// errNotBlocks is returned for content that is neither a string nor blocks.
var errNotBlocks = errors.New("content must be a string or an array of content blocks")

// parseContent normalises the string-or-blocks content shape to blocks.
func parseContent(raw json.RawMessage) ([]Block, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil, nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, err
		}
		return []Block{{Type: "text", Text: s}}, nil
	}
	if raw[0] != '[' {
		return nil, errNotBlocks
	}
	var blocks []Block
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, err
	}
	return blocks, nil
}

// MessagesResponse is a complete (non-streaming) message.
type MessagesResponse struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Model        string          `json:"model"`
	Content      []ResponseBlock `json:"content"`
	StopReason   *string         `json:"stop_reason"`
	StopSequence *string         `json:"stop_sequence"`
	Usage        Usage           `json:"usage"`
}

// ResponseBlock is a content block the translation emits. Separate from Block
// so required-but-empty fields (a thinking signature, a tool input of {}) are
// always present on the wire.
type ResponseBlock struct {
	Type      string          `json:"type"`
	Text      *string         `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	Thinking  *string         `json:"thinking,omitempty"`
	Signature *string         `json:"signature,omitempty"`
}

// Usage is Anthropic token accounting. input_tokens EXCLUDES the cached part.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	OutputTokens             int `json:"output_tokens"`
}

// ---- OpenAI Chat Completions, the backend side ----

type oaRequest struct {
	Model               string         `json:"model"`
	Messages            []oaMessage    `json:"messages"`
	MaxTokens           *int           `json:"max_tokens,omitempty"`
	Temperature         *float64       `json:"temperature,omitempty"`
	TopP                *float64       `json:"top_p,omitempty"`
	TopK                *int           `json:"top_k,omitempty"`
	Stop                []string       `json:"stop,omitempty"`
	Stream              bool           `json:"stream,omitempty"`
	StreamOptions       *oaStreamOpts  `json:"stream_options,omitempty"`
	Tools               []oaTool       `json:"tools,omitempty"`
	ToolChoice          any            `json:"tool_choice,omitempty"`
	ParallelToolCalls   *bool          `json:"parallel_tool_calls,omitempty"`
	User                string         `json:"user,omitempty"`
	ChatTemplateKwargs  map[string]any `json:"chat_template_kwargs,omitempty"`
	SkipSpecialTokens   *bool          `json:"skip_special_tokens,omitempty"`
	IncludeStopStrInOut *bool          `json:"include_stop_str_in_output,omitempty"`
}

type oaStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

type oaMessage struct {
	Role       string       `json:"role"`
	Content    any          `json:"content,omitempty"`
	ToolCalls  []oaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
}

type oaPart struct {
	Type     string      `json:"type"`
	Text     string      `json:"text,omitempty"`
	ImageURL *oaImageURL `json:"image_url,omitempty"`
}

type oaImageURL struct {
	URL string `json:"url"`
}

type oaTool struct {
	Type     string     `json:"type"`
	Function oaFunction `json:"function"`
}

type oaFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type oaToolCall struct {
	Index    *int           `json:"index,omitempty"`
	ID       string         `json:"id,omitempty"`
	Type     string         `json:"type,omitempty"`
	Function oaFunctionCall `json:"function"`
}

type oaFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

// oaResponse covers both a complete response and a stream chunk.
type oaResponse struct {
	ID      string     `json:"id"`
	Model   string     `json:"model"`
	Choices []oaChoice `json:"choices"`
	Usage   *oaUsage   `json:"usage"`
	Error   *oaError   `json:"error"`
}

type oaChoice struct {
	Index        int             `json:"index"`
	Message      *oaOutMessage   `json:"message"`
	Delta        *oaOutMessage   `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
	StopReason   json.RawMessage `json:"stop_reason"`
}

type oaOutMessage struct {
	Role    string  `json:"role"`
	Content *string `json:"content"`
	// Reasoning arrives under either name depending on the vLLM version:
	// reasoning_content (DeepSeek's original field) or reasoning.
	ReasoningContent *string      `json:"reasoning_content"`
	Reasoning        *string      `json:"reasoning"`
	ToolCalls        []oaToolCall `json:"tool_calls"`
}

func (m *oaOutMessage) reasoning() string {
	if m.ReasoningContent != nil && *m.ReasoningContent != "" {
		return *m.ReasoningContent
	}
	if m.Reasoning != nil {
		return *m.Reasoning
	}
	return ""
}

type oaUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// anthropic converts OpenAI usage. Anthropic's input_tokens excludes cache
// reads, OpenAI's prompt_tokens includes them; reporting prompt_tokens as
// input_tokens would double-count every cached token.
func (u *oaUsage) anthropic() Usage {
	if u == nil {
		return Usage{}
	}
	cached := 0
	if u.PromptTokensDetails != nil {
		cached = u.PromptTokensDetails.CachedTokens
	}
	in := u.PromptTokens - cached
	if in < 0 {
		in = 0
	}
	return Usage{InputTokens: in, CacheReadInputTokens: cached, OutputTokens: u.CompletionTokens}
}

type oaError struct {
	Message string          `json:"message"`
	Type    string          `json:"type"`
	Code    json.RawMessage `json:"code"`
}
