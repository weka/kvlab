# Anthropic translation mode (`--anthropic-api`) — testing

How to exercise the router's Anthropic Messages ⇄ Chat Completions translation
offline (mock) and against a real OpenAI-only backend with Claude Code. The user
facing reference (flags, feature matrix, limitations) is
[docs/router.md — Use case 5](../../docs/router.md#use-case-5--anthropic-clients-on-an-openai-only-fleet---anthropic-api).

## Code map

| Where | What |
|---|---|
| `router/internal/anthropicapi/` | The whole translation: `request.go` (Anthropic → OpenAI), `response.go` (whole responses, stop reasons, errors), `stream.go` (SSE converter), `handler.go` (the middleware, count_tokens, debug log) |
| `router/serve/anthropic_api.go` | `AnthropicOptions` → middleware, wrapped around the gateway (inside capture) |
| `cli/command_router.go` | `--anthropic-*` flags |
| `router/internal/testutil/anthropicsse/` | Strict Anthropic event-stream parser + grammar oracle (`Assemble`); use it in any new test |
| `router/internal/mockvllm/chatscript.go` | `X-Mock-Script` scripted replies and `--agentic` |
| `router/serve/anthropic_api_test.go` | Through-the-router tests against mock-vllm |

The middleware sits OUTSIDE the gateway: a `/v1/messages` request is rewritten to
a `/v1/chat/completions` request and passed to the normal handler, whose answer is
translated as it is written. So routing, auth, retries and metrics see a chat
request; the access log shows `path=/v1/chat/completions route=chat`.

## Offline: mock-vllm

Scripted reply per request (header value = compact JSON):

```bash
curl -s :8080/v1/messages -H 'content-type: application/json' \
  -H 'X-Mock-Script: {"reasoning":"r","text":"Checking.","chunk_chars":2,"interleave_tools":true,"tool_calls":[{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"},{"name":"get_weather","arguments":"{\"city\":\"Rome\"}"}]}' \
  -d '{"model":"claude-x","max_tokens":200,"stream":true,"thinking":{"type":"enabled","budget_tokens":100},"messages":[{"role":"user","content":"weather?"}],"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]}'
```

Script fields: `reasoning`, `reasoning_field` (`reasoning_content` default |
`reasoning`), `text`, `tool_calls[{id,name,arguments}]`, `finish_reason`,
`stop_reason`, `chunk_chars` (default 3), `interleave_tools`, `cut_after_chunks`
(stream dies with no finish/[DONE]), `error_status` + `error_message`,
`chunk_delay` (ns). The router forwards the header untouched.

Whole agent loops without scripts: `mock-vllm --agentic` — a request offering
tools gets a call to the first tool (args typed from its schema), and once a tool
result is the last message, a text answer. Requests without tools are unaffected.

```bash
go build -o /tmp/mock-vllm ./router/cmd/mock-vllm && go build -o /tmp/kvlab-local .
/tmp/mock-vllm --instances 2 --base-port 9001 --model-id backend-model --agentic
/tmp/kvlab-local router serve --listen :8080 \
  --backends 'http://127.0.0.1:9001|http://127.0.0.1:9002' \
  --anthropic-api --anthropic-model-map '*=backend-model'
```

Unit + through-router tests: `go test ./router/internal/anthropicapi/ ./router/serve/ -run 'Anthropic|.'`.

## Real backend + Claude Code (acceptance)

```bash
# Router in front of an OpenAI-only endpoint; key read from a file, never a flag
kvlab router serve --listen 127.0.0.1:18080 --metrics-listen 127.0.0.1:29080 \
  --route "* => http://<host>:<port> using <key-file>" \
  --anthropic-api --anthropic-model-map '*=<backend model id from /v1/models>' \
  --anthropic-debug-dir <durable-dir>/debug --log-format text > <durable-dir>/router.log 2>&1

# From inside the repo; READ-ONLY tools only — the model behind it is not Claude
ANTHROPIC_BASE_URL=http://127.0.0.1:18080 ANTHROPIC_AUTH_TOKEN=dummy ANTHROPIC_API_KEY= \
ANTHROPIC_MODEL=claude-sonnet-4-5 ANTHROPIC_SMALL_FAST_MODEL=claude-haiku-4-5 \
claude -p "<question needing many tool calls>" --tools "Read,Grep,Glob" --allowedTools "Read,Grep,Glob,LS" \
  --output-format stream-json --verbose > <durable-dir>/cc.jsonl
```

- **`--tools` is what restricts the model; `--allowedTools` alone does not.**
  `--allowedTools` only pre-approves; every other tool (Bash, Edit, Write, Agent,
  ...) is still offered to the model, and Claude Code auto-approves Bash commands
  it classifies read-only (`ls`, `find`, `wc` ran that way in a first e2e). `--tools`
  removes the rest from the request entirely. Never add
  `--dangerously-skip-permissions`.
- Agent subagents send a `role: "system"` turn inside `messages`; the translation
  maps it to a system message.
- It sends `thinking: {type: enabled, budget_tokens: 31999}`, `max_tokens: 32000`,
  `context_management`, `metadata.user_id`, and probes `HEAD /api/hello` (404 via
  passthrough is harmless).
- Translation bug vs model behaviour: read `<debug-dir>/anthropic-exchanges.jsonl`
  — `openai_response` is the raw backend stream, `anthropic_response` what the
  client got. A malformed tool call in `openai_response` is the model.
- Backend with no `/tokenize` (vLLM Rust frontend): `count_tokens` falls back to
  the byte estimate; a `POST /tokenize 404` in the router log is that probe.
