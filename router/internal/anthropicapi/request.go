package anthropicapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// RequestError is a request the translation refuses. It renders as a 400
// invalid_request_error, which is what Anthropic answers for the same input.
type RequestError struct{ Msg string }

func (e *RequestError) Error() string { return e.Msg }

func badRequest(format string, a ...any) error {
	return &RequestError{Msg: fmt.Sprintf(format, a...)}
}

// translated is a request ready for the backend, plus what the response
// translation needs to remember about the original.
type translated struct {
	body []byte
	// clientModel is echoed in the response: the caller asked for this name,
	// and a client comparing the two must not see the backend's.
	clientModel string
	stream      bool
	// surfaceThinking decides whether backend reasoning becomes thinking
	// blocks or is dropped.
	surfaceThinking bool
	stopSequences   []string
	// dropped names input the backend cannot represent and so never saw.
	dropped []string
}

// toOpenAI translates a Messages request into a Chat Completions request.
func toOpenAI(raw []byte, cfg *Config) (*translated, error) {
	var req MessagesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, badRequest("malformed JSON body: %v", err)
	}
	if req.Model == "" {
		return nil, badRequest("model: field required")
	}
	if len(req.Messages) == 0 {
		return nil, badRequest("messages: at least one message is required")
	}

	out := oaRequest{
		Model:       cfg.mapModel(req.Model),
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
		Stop:        req.StopSequences,
		Stream:      req.Stream,
	}
	tr := &translated{
		clientModel:   req.Model,
		stream:        req.Stream,
		stopSequences: req.StopSequences,
	}
	if req.Stream {
		// Without this a vLLM stream carries no usage at all, and the final
		// message_delta could report none.
		out.StreamOptions = &oaStreamOpts{IncludeUsage: true}
	}
	if req.MaxTokens != nil {
		mt := *req.MaxTokens
		if cfg.MaxTokensCap > 0 && mt > cfg.MaxTokensCap {
			mt = cfg.MaxTokensCap
		}
		out.MaxTokens = &mt
	}
	if req.Metadata != nil && req.Metadata.UserID != "" {
		out.User = req.Metadata.UserID
	}

	// thinking -> the chat template's reasoning switch. Absent leaves the
	// backend's own default, so a request that never mentions thinking is
	// served exactly as the backend would serve it natively.
	if req.Thinking != nil && len(cfg.ThinkingKwargs) > 0 {
		on := req.Thinking.enabled()
		if req.Thinking.Type == "enabled" || req.Thinking.Type == "adaptive" || req.Thinking.Type == "disabled" {
			out.ChatTemplateKwargs = map[string]any{}
			for _, k := range cfg.ThinkingKwargs {
				out.ChatTemplateKwargs[k] = on
			}
		}
	}
	switch cfg.Thinking {
	case ThinkingAlways:
		tr.surfaceThinking = true
	case ThinkingNever:
	default:
		tr.surfaceThinking = req.Thinking.enabled()
	}

	sys, err := systemText(req.System)
	if err != nil {
		return nil, err
	}
	if sys != "" {
		out.Messages = append(out.Messages, oaMessage{Role: "system", Content: sys})
	}

	for i, m := range req.Messages {
		blocks, err := parseContent(m.Content)
		if err != nil {
			return nil, badRequest("messages.%d.content: %v", i, err)
		}
		switch m.Role {
		case "user":
			msgs, dropped, err := userMessages(blocks, i)
			if err != nil {
				return nil, err
			}
			out.Messages = append(out.Messages, msgs...)
			tr.dropped = append(tr.dropped, dropped...)
		case "assistant":
			msg, err := assistantMessage(blocks, i)
			if err != nil {
				return nil, err
			}
			out.Messages = append(out.Messages, msg)
		case "system":
			// Claude Code puts system-role turns inside messages (its subagents
			// do, for one); Chat Completions takes a system message anywhere.
			text, err := systemText(m.Content)
			if err != nil {
				return nil, badRequest("messages.%d.content: %v", i, err)
			}
			out.Messages = append(out.Messages, oaMessage{Role: "system", Content: text})
		default:
			return nil, badRequest("messages.%d.role: must be \"user\", \"assistant\" or \"system\", got %q", i, m.Role)
		}
	}

	for _, t := range req.Tools {
		if !t.isClientTool() {
			tr.dropped = append(tr.dropped, "server tool "+t.Type)
			continue
		}
		if t.Name == "" {
			return nil, badRequest("tools: every tool needs a name")
		}
		params := t.InputSchema
		if len(bytes.TrimSpace(params)) == 0 || bytes.Equal(bytes.TrimSpace(params), []byte("null")) {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out.Tools = append(out.Tools, oaTool{Type: "function", Function: oaFunction{
			Name: t.Name, Description: t.Description, Parameters: params,
		}})
	}
	if req.ToolChoice != nil && len(out.Tools) > 0 {
		switch req.ToolChoice.Type {
		case "", "auto":
			out.ToolChoice = "auto"
		case "any":
			out.ToolChoice = "required"
		case "none":
			out.ToolChoice = "none"
		case "tool":
			if req.ToolChoice.Name == "" {
				return nil, badRequest("tool_choice.name: required when type is \"tool\"")
			}
			out.ToolChoice = map[string]any{"type": "function",
				"function": map[string]string{"name": req.ToolChoice.Name}}
		default:
			return nil, badRequest("tool_choice.type: unknown value %q", req.ToolChoice.Type)
		}
		if req.ToolChoice.DisableParallelToolUse {
			f := false
			out.ParallelToolCalls = &f
		}
	}

	body, err := json.Marshal(out)
	if err != nil {
		return nil, err
	}
	tr.body = body
	return tr, nil
}

// systemText flattens the system prompt. Block form exists so blocks can carry
// cache_control; Chat Completions has no such thing, and the backend's own
// prefix cache needs no hint, so only the text survives.
func systemText(raw json.RawMessage) (string, error) {
	blocks, err := parseContent(raw)
	if err != nil {
		return "", badRequest("system: %v", err)
	}
	parts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		if b.Type == "text" && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n\n"), nil
}

// userMessages translates one user turn. tool_result blocks become role=tool
// messages, which Chat Completions requires to come immediately after the
// assistant message that called them, so they are emitted first; whatever
// else the turn carried follows as one user message.
func userMessages(blocks []Block, i int) ([]oaMessage, []string, error) {
	var out []oaMessage
	var dropped []string
	var parts []oaPart
	for j, b := range blocks {
		switch b.Type {
		case "text":
			parts = append(parts, oaPart{Type: "text", Text: b.Text})
		case "image":
			p, err := imagePart(b.source())
			if err != nil {
				return nil, nil, badRequest("messages.%d.content.%d: %v", i, j, err)
			}
			parts = append(parts, p)
		case "document":
			text, ok := documentText(b)
			if !ok {
				dropped = append(dropped, "document block")
				text = "[a document was attached here, but this backend cannot read documents]"
			}
			parts = append(parts, oaPart{Type: "text", Text: text})
		case "tool_result":
			if b.ToolUseID == "" {
				return nil, nil, badRequest("messages.%d.content.%d.tool_use_id: field required", i, j)
			}
			text, images, err := toolResultContent(b)
			if err != nil {
				return nil, nil, badRequest("messages.%d.content.%d: %v", i, j, err)
			}
			out = append(out, oaMessage{Role: "tool", ToolCallID: b.ToolUseID, Content: text})
			if len(images) > 0 {
				// A tool message carries text only. The image still reaches
				// the model, in the user turn right after the results.
				parts = append(parts, oaPart{Type: "text",
					Text: "Image output of tool call " + b.ToolUseID + ":"})
				parts = append(parts, images...)
			}
		case "thinking", "redacted_thinking":
			// Only meaningful in assistant turns; ignore here as Anthropic does.
		default:
			// search_result, container_upload, ... have no Chat equivalent.
			dropped = append(dropped, b.Type+" block")
		}
	}
	if len(parts) > 0 {
		out = append(out, oaMessage{Role: "user", Content: userContent(parts)})
	} else if len(out) == 0 {
		// An empty user turn still holds its place in the alternation.
		out = append(out, oaMessage{Role: "user", Content: ""})
	}
	return out, dropped, nil
}

// userContent uses the plain string form when the turn is text only: every
// chat template accepts a string, not every one handles a parts array.
func userContent(parts []oaPart) any {
	texts := make([]string, 0, len(parts))
	for _, p := range parts {
		if p.Type != "text" {
			return parts
		}
		texts = append(texts, p.Text)
	}
	return strings.Join(texts, "\n\n")
}

func imagePart(src *Source) (oaPart, error) {
	if src == nil {
		return oaPart{}, fmt.Errorf("image.source: field required")
	}
	switch src.Type {
	case "base64":
		if src.MediaType == "" || src.Data == "" {
			return oaPart{}, fmt.Errorf("image.source: base64 needs media_type and data")
		}
		return oaPart{Type: "image_url", ImageURL: &oaImageURL{
			URL: "data:" + src.MediaType + ";base64," + src.Data}}, nil
	case "url":
		if src.URL == "" {
			return oaPart{}, fmt.Errorf("image.source.url: field required")
		}
		return oaPart{Type: "image_url", ImageURL: &oaImageURL{URL: src.URL}}, nil
	}
	return oaPart{}, fmt.Errorf("image.source.type: unsupported %q", src.Type)
}

// documentText recovers text from a document block when its source is text.
// PDFs and file references cannot be passed to a chat backend.
func documentText(b Block) (string, bool) {
	src := b.source()
	if src == nil {
		return "", false
	}
	var body string
	switch src.Type {
	case "text":
		body = src.Data
	case "content":
		blocks, err := parseContent(src.Content)
		if err != nil {
			return "", false
		}
		var sb strings.Builder
		for _, c := range blocks {
			if c.Type == "text" {
				sb.WriteString(c.Text)
			}
		}
		body = sb.String()
	default:
		return "", false
	}
	if b.Title != "" {
		body = b.Title + "\n\n" + body
	}
	return body, true
}

// toolResultContent flattens a tool_result to the text a tool message carries,
// returning any images separately.
func toolResultContent(b Block) (string, []oaPart, error) {
	blocks, err := parseContent(b.Content)
	if err != nil {
		return "", nil, fmt.Errorf("tool_result.content: %v", err)
	}
	var texts []string
	var images []oaPart
	for _, c := range blocks {
		switch c.Type {
		case "text":
			texts = append(texts, c.Text)
		case "image":
			p, err := imagePart(c.source())
			if err != nil {
				return "", nil, err
			}
			images = append(images, p)
		case "document":
			if t, ok := documentText(c); ok {
				texts = append(texts, t)
			}
		}
	}
	text := strings.Join(texts, "\n")
	if b.IsError {
		// Chat Completions has no error flag on a tool message; say it in the
		// content, where the model will read it.
		text = "Error: " + text
	}
	return text, images, nil
}

// assistantMessage translates one assistant turn: text is concatenated,
// tool_use becomes tool_calls. Thinking blocks are dropped: their signatures
// mean nothing to the backend, and reasoning templates (DeepSeek's among them)
// strip earlier turns' reasoning anyway.
func assistantMessage(blocks []Block, i int) (oaMessage, error) {
	msg := oaMessage{Role: "assistant"}
	var texts []string
	for j, b := range blocks {
		switch b.Type {
		case "text":
			if b.Text != "" {
				texts = append(texts, b.Text)
			}
		case "tool_use", "server_tool_use":
			if b.Type == "server_tool_use" {
				continue
			}
			if b.ID == "" || b.Name == "" {
				return msg, badRequest("messages.%d.content.%d: tool_use needs id and name", i, j)
			}
			args := "{}"
			if in := bytes.TrimSpace(b.Input); len(in) > 0 && !bytes.Equal(in, []byte("null")) {
				args = string(in)
			}
			msg.ToolCalls = append(msg.ToolCalls, oaToolCall{ID: b.ID, Type: "function",
				Function: oaFunctionCall{Name: b.Name, Arguments: args}})
		}
	}
	if len(texts) > 0 {
		msg.Content = strings.Join(texts, "\n\n")
	} else if len(msg.ToolCalls) == 0 {
		msg.Content = ""
	}
	return msg, nil
}
