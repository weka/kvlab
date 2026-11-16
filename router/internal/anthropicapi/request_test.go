package anthropicapi

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

// translate runs toOpenAI and decodes the result generically, so assertions
// read like the wire format.
func translate(t *testing.T, cfg Config, body string) (map[string]any, *translated) {
	t.Helper()
	cfg.withDefaults()
	tr, err := toOpenAI([]byte(body), &cfg)
	if err != nil {
		t.Fatalf("toOpenAI: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(tr.body, &out); err != nil {
		t.Fatalf("translated body is not JSON: %v", err)
	}
	return out, tr
}

func messages(t *testing.T, out map[string]any) []map[string]any {
	t.Helper()
	raw, _ := out["messages"].([]any)
	var ms []map[string]any
	for _, m := range raw {
		ms = append(ms, m.(map[string]any))
	}
	return ms
}

func jsonEq(t *testing.T, got any, want string) {
	t.Helper()
	var w any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("bad want JSON: %v", err)
	}
	g, _ := json.Marshal(got)
	var gn any
	_ = json.Unmarshal(g, &gn)
	if !reflect.DeepEqual(gn, w) {
		t.Errorf("got  %s\nwant %s", g, want)
	}
}

func TestSystemBlocksWithCacheControlFlattenToOneSystemMessage(t *testing.T) {
	out, _ := translate(t, Config{}, `{"model":"m","max_tokens":10,
	  "system":[{"type":"text","text":"You are helpful.","cache_control":{"type":"ephemeral"}},
	            {"type":"text","text":"Be brief.","cache_control":{"type":"ephemeral","ttl":"1h"}}],
	  "messages":[{"role":"user","content":"hi"}]}`)
	ms := messages(t, out)
	jsonEq(t, ms[0], `{"role":"system","content":"You are helpful.\n\nBe brief."}`)
	jsonEq(t, ms[1], `{"role":"user","content":"hi"}`)
}

func TestStringSystemPrompt(t *testing.T) {
	out, _ := translate(t, Config{}, `{"model":"m","max_tokens":10,"system":"sys",
	  "messages":[{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}},{"type":"text","text":"b"}]}]}`)
	ms := messages(t, out)
	jsonEq(t, ms, `[{"role":"system","content":"sys"},{"role":"user","content":"a\n\nb"}]`)
}

func TestScalarParamsAndStreamUsage(t *testing.T) {
	out, tr := translate(t, Config{}, `{"model":"claude-x","max_tokens":321,"temperature":0.2,"top_p":0.9,"top_k":40,
	  "stop_sequences":["END","STOP"],"stream":true,"metadata":{"user_id":"u-1"},
	  "messages":[{"role":"user","content":"hi"}]}`)
	for k, want := range map[string]any{"model": "claude-x", "max_tokens": 321.0, "temperature": 0.2,
		"top_p": 0.9, "top_k": 40.0, "stream": true, "user": "u-1"} {
		if !reflect.DeepEqual(out[k], want) {
			t.Errorf("%s = %v, want %v", k, out[k], want)
		}
	}
	jsonEq(t, out["stop"], `["END","STOP"]`)
	jsonEq(t, out["stream_options"], `{"include_usage":true}`)
	if !tr.stream || tr.clientModel != "claude-x" {
		t.Errorf("translated meta = %+v", tr)
	}
}

func TestModelMapFirstMatchWinsAndUnmatchedPassesThrough(t *testing.T) {
	var cfg Config
	for _, s := range []string{"haiku=small-model", "claude=big-model"} {
		r, err := ParseModelRule(s)
		if err != nil {
			t.Fatal(err)
		}
		cfg.ModelMap = append(cfg.ModelMap, r)
	}
	for model, want := range map[string]string{
		"claude-haiku-4-5":  "small-model",
		"claude-sonnet-4-5": "big-model",
		"other":             "other",
	} {
		out, tr := translate(t, cfg, `{"model":"`+model+`","max_tokens":1,"messages":[{"role":"user","content":"x"}]}`)
		if out["model"] != want || tr.clientModel != model {
			t.Errorf("%s -> %v (client %s), want %s", model, out["model"], tr.clientModel, want)
		}
	}
	star, _ := ParseModelRule("*=everything")
	out, _ := translate(t, Config{ModelMap: []ModelRule{star}}, `{"model":"claude-opus-4","max_tokens":1,"messages":[{"role":"user","content":"x"}]}`)
	if out["model"] != "everything" {
		t.Errorf("* map: %v", out["model"])
	}
	for _, bad := range []string{"", "=x", "x=", "noequals"} {
		if _, err := ParseModelRule(bad); err == nil {
			t.Errorf("ParseModelRule(%q) accepted", bad)
		}
	}
}

func TestMaxTokensCap(t *testing.T) {
	out, _ := translate(t, Config{MaxTokensCap: 1000}, `{"model":"m","max_tokens":32000,"messages":[{"role":"user","content":"x"}]}`)
	if out["max_tokens"] != 1000.0 {
		t.Errorf("max_tokens = %v, want 1000", out["max_tokens"])
	}
	out, _ = translate(t, Config{MaxTokensCap: 1000}, `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}]}`)
	if out["max_tokens"] != 10.0 {
		t.Errorf("max_tokens = %v, want 10", out["max_tokens"])
	}
}

func TestImagesBase64AndURL(t *testing.T) {
	out, _ := translate(t, Config{}, `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
	  {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}},
	  {"type":"image","source":{"type":"url","url":"https://example.com/a.jpg"}},
	  {"type":"text","text":"what is this?"}]}]}`)
	jsonEq(t, messages(t, out)[0], `{"role":"user","content":[
	  {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},
	  {"type":"image_url","image_url":{"url":"https://example.com/a.jpg"}},
	  {"type":"text","text":"what is this?"}]}`)
}

// A full tool round trip: assistant tool_use -> tool_calls; user tool_result
// (string, blocks, is_error, with an image) -> role=tool messages first, then
// the rest of the turn.
func TestToolUseAndToolResultRoundTrip(t *testing.T) {
	out, _ := translate(t, Config{}, `{"model":"m","max_tokens":1,"messages":[
	  {"role":"user","content":"read two files"},
	  {"role":"assistant","content":[
	    {"type":"thinking","thinking":"plan","signature":"sig"},
	    {"type":"text","text":"Reading."},
	    {"type":"tool_use","id":"toolu_1","name":"Read","input":{"file_path":"/a"}},
	    {"type":"tool_use","id":"toolu_2","name":"Read","input":{}}]},
	  {"role":"user","content":[
	    {"type":"tool_result","tool_use_id":"toolu_1","content":"contents of a"},
	    {"type":"tool_result","tool_use_id":"toolu_2","is_error":true,
	     "content":[{"type":"text","text":"no such file"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"BB"}}]},
	    {"type":"text","text":"now summarise"}]}]}`)
	jsonEq(t, messages(t, out), `[
	  {"role":"user","content":"read two files"},
	  {"role":"assistant","content":"Reading.","tool_calls":[
	    {"id":"toolu_1","type":"function","function":{"name":"Read","arguments":"{\"file_path\":\"/a\"}"}},
	    {"id":"toolu_2","type":"function","function":{"name":"Read","arguments":"{}"}}]},
	  {"role":"tool","tool_call_id":"toolu_1","content":"contents of a"},
	  {"role":"tool","tool_call_id":"toolu_2","content":"Error: no such file"},
	  {"role":"user","content":[
	    {"type":"text","text":"Image output of tool call toolu_2:"},
	    {"type":"image_url","image_url":{"url":"data:image/png;base64,BB"}},
	    {"type":"text","text":"now summarise"}]}]`)
}

func TestAssistantWithOnlyToolCallsHasNoContent(t *testing.T) {
	out, _ := translate(t, Config{}, `{"model":"m","max_tokens":1,"messages":[
	  {"role":"user","content":"x"},
	  {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}]},
	  {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1"}]}]}`)
	ms := messages(t, out)
	if _, has := ms[1]["content"]; has {
		t.Errorf("assistant message with only tool calls carries content: %v", ms[1])
	}
	jsonEq(t, ms[2], `{"role":"tool","tool_call_id":"t1","content":""}`)
	if len(ms) != 3 {
		t.Errorf("a turn of tool results only must not add an empty user message: %v", ms)
	}
}

func TestToolsAndToolChoice(t *testing.T) {
	tools := `"tools":[{"name":"get","description":"d","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}},
	  {"name":"noschema"},
	  {"type":"web_search_20250305","name":"web_search","max_uses":3}]`
	cases := map[string]string{
		`{"type":"auto"}`:              `"auto"`,
		`{"type":"any"}`:               `"required"`,
		`{"type":"none"}`:              `"none"`,
		`{"type":"tool","name":"get"}`: `{"type":"function","function":{"name":"get"}}`,
	}
	for choice, want := range cases {
		out, tr := translate(t, Config{}, `{"model":"m","max_tokens":1,`+tools+`,"tool_choice":`+choice+`,
		  "messages":[{"role":"user","content":"x"}]}`)
		jsonEq(t, out["tool_choice"], want)
		jsonEq(t, out["tools"], `[
		  {"type":"function","function":{"name":"get","description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}},
		  {"type":"function","function":{"name":"noschema","parameters":{"type":"object","properties":{}}}}]`)
		if len(tr.dropped) != 1 || !strings.Contains(tr.dropped[0], "web_search") {
			t.Errorf("server tool not reported dropped: %v", tr.dropped)
		}
		if _, has := out["parallel_tool_calls"]; has {
			t.Errorf("parallel_tool_calls set without disable_parallel_tool_use")
		}
	}
	out, _ := translate(t, Config{}, `{"model":"m","max_tokens":1,`+tools+`,
	  "tool_choice":{"type":"auto","disable_parallel_tool_use":true},"messages":[{"role":"user","content":"x"}]}`)
	if out["parallel_tool_calls"] != false {
		t.Errorf("parallel_tool_calls = %v, want false", out["parallel_tool_calls"])
	}
}

func TestThinkingMapsToTemplateKwargs(t *testing.T) {
	cfg := Config{ThinkingKwargs: []string{"thinking", "enable_thinking"}}
	out, tr := translate(t, cfg, `{"model":"m","max_tokens":100,"thinking":{"type":"enabled","budget_tokens":50},
	  "messages":[{"role":"user","content":"x"}]}`)
	jsonEq(t, out["chat_template_kwargs"], `{"thinking":true,"enable_thinking":true}`)
	if !tr.surfaceThinking {
		t.Error("thinking enabled but reasoning would not be surfaced")
	}
	out, tr = translate(t, cfg, `{"model":"m","max_tokens":100,"thinking":{"type":"disabled"},
	  "messages":[{"role":"user","content":"x"}]}`)
	jsonEq(t, out["chat_template_kwargs"], `{"thinking":false,"enable_thinking":false}`)
	if tr.surfaceThinking {
		t.Error("thinking disabled but reasoning would be surfaced")
	}
	out, _ = translate(t, cfg, `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"x"}]}`)
	if _, has := out["chat_template_kwargs"]; has {
		t.Error("no thinking in the request must leave the backend default alone")
	}
	out, _ = translate(t, Config{}, `{"model":"m","max_tokens":100,"thinking":{"type":"enabled","budget_tokens":50},
	  "messages":[{"role":"user","content":"x"}]}`)
	if _, has := out["chat_template_kwargs"]; has {
		t.Error("no configured kwargs must send none")
	}
	_, tr = translate(t, Config{Thinking: ThinkingAlways}, `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}]}`)
	if !tr.surfaceThinking {
		t.Error("ThinkingAlways")
	}
	_, tr = translate(t, Config{Thinking: ThinkingNever}, `{"model":"m","max_tokens":1,"thinking":{"type":"enabled","budget_tokens":5},"messages":[{"role":"user","content":"x"}]}`)
	if tr.surfaceThinking {
		t.Error("ThinkingNever")
	}
}

// Fields Claude Code sends that have no Chat Completions meaning must be
// tolerated, not refused.
func TestUnknownAndAnthropicOnlyFieldsAreTolerated(t *testing.T) {
	out, tr := translate(t, Config{}, `{"model":"m","max_tokens":1,
	  "context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},
	  "service_tier":"auto","container":"c","mcp_servers":[],
	  "thinking":{"type":"enabled","budget_tokens":10,"display":"updates"},
	  "messages":[{"role":"user","content":[
	    {"type":"text","text":"x","cache_control":{"type":"ephemeral"},"citations":[]},
	    {"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc body"},"title":"T"},
	    {"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBER"}},
	    {"type":"search_result","source":"s","title":"t","content":[]}]}]}`)
	for _, k := range []string{"context_management", "service_tier", "container", "mcp_servers"} {
		if _, has := out[k]; has {
			t.Errorf("%s forwarded to the backend", k)
		}
	}
	got := messages(t, out)[0]["content"].(string)
	if !strings.Contains(got, "T\n\ndoc body") || !strings.Contains(got, "cannot read documents") {
		t.Errorf("documents: %q", got)
	}
	if len(tr.dropped) != 2 {
		t.Errorf("dropped = %v, want the PDF and the search_result", tr.dropped)
	}
}

func TestInvalidRequestsAreRefused(t *testing.T) {
	for name, body := range map[string]string{
		"not json":          `{`,
		"no model":          `{"max_tokens":1,"messages":[{"role":"user","content":"x"}]}`,
		"no messages":       `{"model":"m","max_tokens":1,"messages":[]}`,
		"bad role":          `{"model":"m","max_tokens":1,"messages":[{"role":"system","content":"x"}]}`,
		"bad content":       `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":5}]}`,
		"tool_use no id":    `{"model":"m","max_tokens":1,"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"f","input":{}}]}]}`,
		"result no id":      `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"tool_result","content":"x"}]}]}`,
		"bad image":         `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"file","file_id":"f"}}]}]}`,
		"choice no name":    `{"model":"m","max_tokens":1,"tools":[{"name":"f","input_schema":{}}],"tool_choice":{"type":"tool"},"messages":[{"role":"user","content":"x"}]}`,
		"choice unknown":    `{"model":"m","max_tokens":1,"tools":[{"name":"f","input_schema":{}}],"tool_choice":{"type":"sometimes"},"messages":[{"role":"user","content":"x"}]}`,
		"tool without name": `{"model":"m","max_tokens":1,"tools":[{"input_schema":{}}],"messages":[{"role":"user","content":"x"}]}`,
	} {
		cfg := Config{}
		cfg.withDefaults()
		_, err := toOpenAI([]byte(body), &cfg)
		var re *RequestError
		if !errors.As(err, &re) {
			t.Errorf("%s: err = %v, want a RequestError", name, err)
		}
	}
}
