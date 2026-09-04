package openai

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/betterlmy/kiro-proxy/internal/anthropic"
	"github.com/betterlmy/kiro-proxy/internal/respconv"
)

func TestNormalizeResponseTools_CodexToolSet(t *testing.T) {
	tools := []ResponseTool{
		{Type: "function", Name: "read_file", Description: "Read a file", Parameters: map[string]any{"type": "object"}},
		{Type: "custom", Name: "apply_patch", Description: "Apply a patch"},
		{
			Type: "namespace", Name: "mcp__pencil",
			Tools: []ResponseTool{{Type: "function", Name: "execute", Parameters: map[string]any{"type": "object"}}},
		},
		{
			Type: "namespace", Name: "multi_agent_v1",
			Tools: []ResponseTool{{Type: "function", Name: "spawn_agent", Parameters: map[string]any{"type": "object"}}},
		},
		{Type: "web_search"},
	}

	got, kinds, err := normalizeResponseTools(tools)
	if err != nil {
		t.Fatal(err)
	}
	wantNames := []string{"read_file", "apply_patch", "mcp__pencil__execute", "multi_agent_v1__spawn_agent"}
	if len(got) != len(wantNames) {
		t.Fatalf("tool count = %d, want %d", len(got), len(wantNames))
	}
	for i, want := range wantNames {
		if got[i].Name != want {
			t.Fatalf("tool[%d].Name = %q, want %q", i, got[i].Name, want)
		}
	}
	if kinds["apply_patch"] != responseToolKindCustom {
		t.Fatalf("apply_patch kind = %q, want custom", kinds["apply_patch"])
	}
	schema := got[1].InputSchema
	if schema["type"] != "object" {
		t.Fatalf("custom schema type = %v, want object", schema["type"])
	}
	properties, _ := schema["properties"].(map[string]any)
	if _, ok := properties["input"]; !ok {
		t.Fatalf("custom schema = %#v, missing input property", schema)
	}
}

func TestNormalizeResponseTools_RejectsUnknownType(t *testing.T) {
	_, _, err := normalizeResponseTools([]ResponseTool{{Type: "computer"}})
	if err == nil || !strings.Contains(err.Error(), "unsupported tool type") {
		t.Fatalf("error = %v, want unsupported tool type", err)
	}
}

func TestResponseInputMessages_CustomToolRoundTrip(t *testing.T) {
	messages, err := responseInputMessages([]any{
		map[string]any{"type": "custom_tool_call", "call_id": "call_patch", "name": "apply_patch", "input": "*** Begin Patch"},
		map[string]any{"type": "custom_tool_call_output", "call_id": "call_patch", "output": "Done"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 2 {
		t.Fatalf("message count = %d, want 2", len(messages))
	}
	use := messages[0].Content.Blocks[0]
	if use.Name != "apply_patch" || use.Input["input"] != "*** Begin Patch" {
		t.Fatalf("custom tool use = %#v", use)
	}
	result := messages[1].Content.Blocks[0]
	if result.ToolUseID != "call_patch" || result.Content.Text != "Done" {
		t.Fatalf("custom tool result = %#v", result)
	}
}

func TestResponseInputMessages_GroupsParallelToolRound(t *testing.T) {
	messages, err := responseInputMessages([]any{
		map[string]any{"type": "message", "role": "user", "content": "use both tools"},
		map[string]any{"type": "function_call", "call_id": "call_1", "name": "first", "arguments": `{}`},
		map[string]any{"type": "function_call", "call_id": "call_2", "name": "second", "arguments": `{}`},
		map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "first result"},
		map[string]any{"type": "function_call_output", "call_id": "call_2", "output": "second result"},
		map[string]any{"type": "message", "role": "user", "content": "continue"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 3 {
		t.Fatalf("message count = %d, want 3", len(messages))
	}
	if got := len(messages[1].Content.Blocks); got != 2 {
		t.Fatalf("assistant tool uses = %d, want 2", got)
	}
	if got := len(messages[2].Content.Blocks); got != 3 {
		t.Fatalf("user blocks = %d, want 3", got)
	}
	if messages[2].Content.Blocks[0].Type != anthropic.BlockTypeToolResult || messages[2].Content.Blocks[1].Type != anthropic.BlockTypeToolResult || messages[2].Content.Blocks[2].Text != "continue" {
		t.Fatalf("user blocks = %#v", messages[2].Content.Blocks)
	}
}

func TestChatRequestNormalize_GroupsParallelToolRound(t *testing.T) {
	var input ChatRequest
	if err := json.Unmarshal([]byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[
			{"role":"user","content":"use both tools"},
			{"role":"assistant","content":null,"tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"first","arguments":"{}"}},
				{"id":"call_2","type":"function","function":{"name":"second","arguments":"{}"}}
			]},
			{"role":"tool","tool_call_id":"call_1","content":"first result"},
			{"role":"tool","tool_call_id":"call_2","content":"second result"},
			{"role":"user","content":"continue"}
		]
	}`), &input); err != nil {
		t.Fatal(err)
	}

	request, err := input.Normalize()
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 3 {
		t.Fatalf("message count = %d, want 3", len(request.Messages))
	}
	if got := len(request.Messages[1].Content.Blocks); got != 2 {
		t.Fatalf("assistant tool uses = %d, want 2", got)
	}
	userBlocks := request.Messages[2].Content.Blocks
	if got := len(userBlocks); got != 3 {
		t.Fatalf("user blocks = %d, want 3", got)
	}
	if userBlocks[0].ToolUseID != "call_1" || userBlocks[1].ToolUseID != "call_2" || userBlocks[2].Text != "continue" {
		t.Fatalf("user blocks = %#v", userBlocks)
	}
}

func TestOpenAIRequestNormalize_RejectsIncompleteToolCalls(t *testing.T) {
	tests := []struct {
		name string
		run  func() error
		want string
	}{
		{
			name: "chat missing id",
			run: func() error {
				var input ChatRequest
				if err := json.Unmarshal([]byte(`{"model":"model","messages":[{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"read","arguments":"{}"}}]}]}`), &input); err != nil {
					return err
				}
				_, err := input.Normalize()
				return err
			},
			want: "requires id",
		},
		{
			name: "chat missing function name",
			run: func() error {
				var input ChatRequest
				if err := json.Unmarshal([]byte(`{"model":"model","messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"arguments":"{}"}}]}]}`), &input); err != nil {
					return err
				}
				_, err := input.Normalize()
				return err
			},
			want: "requires function name",
		},
		{
			name: "responses missing call id",
			run: func() error {
				_, err := responseInputMessages([]any{map[string]any{"type": "function_call", "name": "read", "arguments": `{}`}})
				return err
			},
			want: "requires call_id and name",
		},
		{
			name: "responses missing name",
			run: func() error {
				_, err := responseInputMessages([]any{map[string]any{"type": "function_call", "call_id": "call_1", "arguments": `{}`}})
				return err
			},
			want: "requires call_id and name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.run()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestResponseOutput_CustomTool(t *testing.T) {
	source := map[string]any{"content": []any{
		map[string]any{"type": "tool_use", "id": "call_patch", "name": "apply_patch", "input": map[string]any{"input": "*** Begin Patch"}},
	}}
	output, err := responseOutput(source, map[string]responseToolKind{"apply_patch": responseToolKindCustom})
	if err != nil {
		t.Fatal(err)
	}
	call := output[0].(map[string]any)
	if call["type"] != "custom_tool_call" || call["input"] != "*** Begin Patch" {
		t.Fatalf("custom response output = %#v", call)
	}
}

func TestResponseOutput_CustomToolRequiresInputString(t *testing.T) {
	source := map[string]any{"content": []any{
		map[string]any{"type": "tool_use", "id": "call_patch", "name": "apply_patch", "input": map[string]any{"patch": "*** Begin Patch"}},
	}}
	_, err := responseOutput(source, map[string]responseToolKind{"apply_patch": responseToolKindCustom})
	if err == nil || !strings.Contains(err.Error(), "must return an input string") {
		t.Fatalf("error = %v, want missing custom input error", err)
	}
}

func TestStreamWriter_CustomToolCall(t *testing.T) {
	recorder := httptest.NewRecorder()
	writer := newStreamWriter(recorder, recorder, surfaceResponses, "resp_1", "model", "", map[string]responseToolKind{"apply_patch": responseToolKindCustom})
	writer.start()
	writer.delta(respconv.EventDelta{ToolStop: true, ToolUseID: "call_patch", ToolName: "apply_patch", ToolInput: `{"input":"*** Begin Patch"}`})

	body := recorder.Body.String()
	for _, want := range []string{"response.custom_tool_call_input.delta", "response.custom_tool_call_input.done", `"type":"custom_tool_call"`, "*** Begin Patch"} {
		if !strings.Contains(body, want) {
			t.Fatalf("stream body missing %q: %s", want, body)
		}
	}
}

func TestStreamWriter_MessageDoneIncludesContent(t *testing.T) {
	t.Run("工具调用前的消息使用空内容数组", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		writer := newStreamWriter(recorder, recorder, surfaceResponses, "resp_1", "model", "", nil)
		writer.delta(respconv.EventDelta{TextDelta: "先说明"})
		writer.delta(respconv.EventDelta{ToolStop: true, ToolUseID: "call_1", ToolName: "read_file", ToolInput: `{}`})

		items := responseMessageDoneItems(t, recorder.Body.String())
		if len(items) != 1 {
			t.Fatalf("message done count = %d, want 1", len(items))
		}
		content, ok := items[0]["content"].([]any)
		if !ok || len(content) != 0 {
			t.Fatalf("message content = %#v, want empty array", items[0]["content"])
		}
	})

	t.Run("收尾消息包含实际文本内容", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		writer := newStreamWriter(recorder, recorder, surfaceResponses, "resp_1", "model", "", nil)
		writer.delta(respconv.EventDelta{TextDelta: "完成"})
		writer.finish(map[string]any{"content": []any{map[string]any{"type": "text", "text": "完成"}}}, false)

		items := responseMessageDoneItems(t, recorder.Body.String())
		if len(items) != 1 {
			t.Fatalf("message done count = %d, want 1", len(items))
		}
		content := items[0]["content"].([]any)
		part := content[0].(map[string]any)
		if part["type"] != "output_text" || part["text"] != "完成" {
			t.Fatalf("message content part = %#v", part)
		}
	})
}

func responseMessageDoneItems(t *testing.T, body string) []map[string]any {
	t.Helper()
	var items []map[string]any
	for line := range strings.Lines(body) {
		data, ok := strings.CutPrefix(strings.TrimSpace(line), "data: ")
		if !ok {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			t.Fatal(err)
		}
		if event["type"] != "response.output_item.done" {
			continue
		}
		item, _ := event["item"].(map[string]any)
		if item["type"] == "message" {
			items = append(items, item)
		}
	}
	return items
}
