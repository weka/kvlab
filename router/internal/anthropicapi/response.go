package anthropicapi

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// stopReason maps an OpenAI finish_reason to Anthropic's stop_reason.
//
// "stop" is ambiguous in Chat Completions: it covers both a natural end and a
// stop string. vLLM disambiguates in its own stop_reason field, which names the
// matched string; when that is one of the caller's stop_sequences, the answer
// is stop_sequence and the sequence is reported.
func stopReason(finish string, vllmStop json.RawMessage, stops []string, sawTools bool) (string, *string) {
	switch finish {
	case "length":
		return "max_tokens", nil
	case "tool_calls", "function_call":
		return "tool_use", nil
	case "content_filter":
		return "refusal", nil
	}
	if sawTools {
		// Some backends finish a tool-calling turn with "stop".
		return "tool_use", nil
	}
	if len(vllmStop) > 0 && vllmStop[0] == '"' {
		var s string
		if json.Unmarshal(vllmStop, &s) == nil {
			for _, want := range stops {
				if s == want {
					return "stop_sequence", &s
				}
			}
		}
	}
	return "end_turn", nil
}

// messageID derives a msg_ id from the backend's, so one exchange can be
// found in both formats' logs.
func messageID(upstream string) string {
	id := strings.TrimPrefix(upstream, "chatcmpl-")
	if id == "" {
		id = randomID()
	}
	return "msg_" + id
}

func randomID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// toolUseID keeps the backend's call id when it is usable. The id round-trips:
// the client sends it back as tool_result.tool_use_id, which becomes the tool
// message's tool_call_id, which the backend matches against its own call.
func toolUseID(id string) string {
	if id == "" {
		return "toolu_" + randomID()
	}
	return id
}

// toolInput turns streamed-or-whole argument text into a tool_use input.
// Anthropic requires an object; anything else is a malformed call from the
// model, reported to the client as an empty input rather than a broken
// response, and logged by the caller.
func toolInput(args string) (json.RawMessage, bool) {
	t := strings.TrimSpace(args)
	if t == "" {
		return json.RawMessage("{}"), true
	}
	if json.Valid([]byte(t)) && t[0] == '{' {
		return json.RawMessage(t), true
	}
	return json.RawMessage("{}"), false
}

func strPtr(s string) *string { return &s }

// fromOpenAI translates a complete Chat Completions response.
func fromOpenAI(body []byte, tr *translated) (*MessagesResponse, []string, error) {
	var resp oaResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, nil, fmt.Errorf("backend response is not JSON: %v", err)
	}
	if len(resp.Choices) == 0 || resp.Choices[0].Message == nil {
		return nil, nil, fmt.Errorf("backend response has no choices")
	}
	ch := resp.Choices[0]
	m := ch.Message
	out := &MessagesResponse{
		ID:      messageID(resp.ID),
		Type:    "message",
		Role:    "assistant",
		Model:   tr.clientModel,
		Content: []ResponseBlock{},
		Usage:   resp.Usage.anthropic(),
	}
	var warnings []string
	if r := m.reasoning(); r != "" && tr.surfaceThinking {
		out.Content = append(out.Content, ResponseBlock{Type: "thinking",
			Thinking: strPtr(r), Signature: strPtr("")})
	}
	if m.Content != nil && *m.Content != "" {
		out.Content = append(out.Content, ResponseBlock{Type: "text", Text: strPtr(*m.Content)})
	}
	for _, tc := range m.ToolCalls {
		in, ok := toolInput(tc.Function.Arguments)
		if !ok {
			warnings = append(warnings, "tool call "+tc.Function.Name+" had non-object arguments: "+
				truncate(tc.Function.Arguments, 200))
		}
		out.Content = append(out.Content, ResponseBlock{Type: "tool_use",
			ID: toolUseID(tc.ID), Name: tc.Function.Name, Input: in})
	}
	finish := ""
	if ch.FinishReason != nil {
		finish = *ch.FinishReason
	}
	sr, seq := stopReason(finish, ch.StopReason, tr.stopSequences, len(m.ToolCalls) > 0)
	out.StopReason = &sr
	out.StopSequence = seq
	return out, warnings, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// ---- errors ----

type errorEnvelope struct {
	Type  string    `json:"type"`
	Error errorBody `json:"error"`
}

type errorBody struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// errorType is Anthropic's error type for an HTTP status.
func errorType(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusServiceUnavailable, 529:
		// A router with nothing healthy, or a backend shedding load: the
		// caller should back off and retry, which is what overloaded_error
		// tells an Anthropic client to do.
		return "overloaded_error"
	}
	if status >= 500 {
		return "api_error"
	}
	return "invalid_request_error"
}

func errorJSON(status int, msg string) []byte {
	b, _ := json.Marshal(errorEnvelope{Type: "error",
		Error: errorBody{Type: errorType(status), Message: msg}})
	return b
}

// writeError renders an Anthropic error response.
func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Del("Content-Length")
	w.WriteHeader(status)
	_, _ = w.Write(errorJSON(status, msg))
}

// upstreamErrorMessage pulls the human-readable message out of whatever error
// body arrived: an OpenAI envelope, a bare {"message":...}, or plain text.
func upstreamErrorMessage(body []byte) string {
	var env struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
		Detail  any             `json:"detail"`
	}
	if json.Unmarshal(body, &env) == nil {
		if len(env.Error) > 0 {
			var e oaError
			if json.Unmarshal(env.Error, &e) == nil && e.Message != "" {
				return e.Message
			}
			var s string
			if json.Unmarshal(env.Error, &s) == nil && s != "" {
				return s
			}
		}
		if env.Message != "" {
			return env.Message
		}
		if env.Detail != nil {
			b, _ := json.Marshal(env.Detail)
			return string(b)
		}
	}
	return truncate(string(bytes.TrimSpace(body)), 2000)
}

// translateErrorMessage rewords the one backend error a client acts on by its
// text. Claude Code reacts to "prompt is too long" by compacting the
// conversation; vLLM's context-length refusal says the same thing differently
// and would otherwise surface as a dead end.
func translateErrorMessage(status int, msg string) string {
	if status == http.StatusBadRequest {
		l := strings.ToLower(msg)
		if strings.Contains(l, "maximum context length") ||
			strings.Contains(l, "context length") && strings.Contains(l, "token") {
			return "prompt is too long: " + msg
		}
	}
	return msg
}
