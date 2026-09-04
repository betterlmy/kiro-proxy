package openai

import (
	"encoding/json"
	"net/http/httptest"
	"os"
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
	t.Run("工具调用前的消息保留已发送文本", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		writer := newStreamWriter(recorder, recorder, surfaceResponses, "resp_1", "model", "", nil)
		writer.delta(respconv.EventDelta{TextDelta: "先说明"})
		writer.delta(respconv.EventDelta{ToolStop: true, ToolUseID: "call_1", ToolName: "read_file", ToolInput: `{}`})

		items := responseMessageDoneItems(t, recorder.Body.String())
		if len(items) != 1 {
			t.Fatalf("message done count = %d, want 1", len(items))
		}
		content, ok := items[0]["content"].([]any)
		if !ok || len(content) != 1 {
			t.Fatalf("message content = %#v, want one text part", items[0]["content"])
		}
		part, _ := content[0].(map[string]any)
		if part["type"] != "output_text" || part["text"] != "先说明" {
			t.Fatalf("message content part = %#v", part)
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

func TestOpenAIContent_InlineImageAndToolResult(t *testing.T) {
	content, err := openAIContent([]any{
		map[string]any{"type": "input_text", "text": "describe this"},
		map[string]any{"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(content.Blocks) != 2 || content.Blocks[1].Type != anthropic.BlockTypeImage {
		t.Fatalf("content = %#v", content)
	}
	if content.Blocks[1].Source.MediaType != "image/png" || content.Blocks[1].Source.Data != "aGVsbG8=" {
		t.Fatalf("image source = %#v", content.Blocks[1].Source)
	}

	messages, err := responseInputMessages([]any{
		map[string]any{"type": "function_call", "call_id": "call.image", "name": "read_image", "arguments": `{}`},
		map[string]any{"type": "function_call_output", "call_id": "call.image", "output": []any{
			map[string]any{"type": "output_text", "text": "found image"},
			map[string]any{"type": "input_image", "image_url": "data:image/webp;base64,aGVsbG8="},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	result := messages[1].Content.Blocks[0]
	if strings.Contains(result.ToolUseID, ".") || len(result.Content.Blocks) != 2 || result.Content.Blocks[1].Type != anthropic.BlockTypeImage {
		t.Fatalf("tool result = %#v", result)
	}

	if _, err := openAIContent([]any{map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/image.png"}}}); err == nil || !strings.Contains(err.Error(), "remote image URLs") {
		t.Fatalf("remote image error = %v", err)
	}
}

func TestResponseInput_NormalizesReasoningAndDuplicateToolResults(t *testing.T) {
	messages, err := responseInputMessages([]any{
		map[string]any{"type": "reasoning", "encrypted_content": "opaque-kiro-state"},
		map[string]any{"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": `{}`},
		map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "partial"},
		map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "final"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 2 {
		t.Fatalf("message count = %d, want 2", len(messages))
	}
	assistant := messages[0].Content.Blocks
	if len(assistant) != 2 || assistant[0].Type != anthropic.BlockTypeRedactedThinking || assistant[1].ID != "call_1" {
		t.Fatalf("assistant blocks = %#v", assistant)
	}
	results := messages[1].Content.Blocks
	if len(results) != 1 || results[0].Content.Text != "final" {
		t.Fatalf("tool results = %#v", results)
	}
}

func TestNormalizeToolChoiceAndAdditionalTools(t *testing.T) {
	chat := ChatRequest{Model: "model", Messages: []ChatMessage{{Role: "user", Content: "hello"}}, ToolChoice: "none"}
	tool := ChatTool{Type: "function"}
	tool.Function.Name = "read_file"
	chat.Tools = []ChatTool{tool}
	normalizedChat, err := chat.Normalize()
	if err != nil {
		t.Fatal(err)
	}
	if normalizedChat.ToolChoice != "none" || len(normalizedChat.Tools) != 0 {
		t.Fatalf("chat tools = %#v, choice = %#v", normalizedChat.Tools, normalizedChat.ToolChoice)
	}
	chat.ToolChoice = "required"
	if _, err := chat.Normalize(); err == nil || !strings.Contains(err.Error(), "cannot be enforced") {
		t.Fatalf("required choice error = %v", err)
	}

	request := ResponsesRequest{
		Model: "model",
		Input: []any{map[string]any{
			"type": "message", "role": "user", "content": "hello",
			"additional_tools": []any{map[string]any{"type": "function", "name": "read_file", "parameters": map[string]any{"type": "object"}}},
		}},
	}
	normalizedResponse, err := request.Normalize()
	if err != nil {
		t.Fatal(err)
	}
	if len(normalizedResponse.Tools) != 1 || normalizedResponse.Tools[0].Name != "read_file" {
		t.Fatalf("additional tools = %#v", normalizedResponse.Tools)
	}
}

func TestResponseOutput_ExposesOpaqueReasoning(t *testing.T) {
	output, err := responseOutput(map[string]any{"content": []any{
		map[string]any{"type": "thinking", "thinking": "considered"},
		map[string]any{"type": "redacted_thinking", "data": "opaque-kiro-state"},
	}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	item := output[0].(map[string]any)
	if item["encrypted_content"] != "opaque-kiro-state" {
		t.Fatalf("reasoning item = %#v", item)
	}
}

func TestStreamWriter_ResponsesLifecycleAndFailure(t *testing.T) {
	recorder := httptest.NewRecorder()
	writer := newStreamWriter(recorder, recorder, surfaceResponses, "resp_1", "model", "", nil)
	writer.start()
	writer.delta(respconv.EventDelta{ThinkingDelta: "think"})
	writer.delta(respconv.EventDelta{TextDelta: "answer"})
	writer.delta(respconv.EventDelta{RedactedContent: "opaque"})
	writer.finish(map[string]any{"content": []any{
		map[string]any{"type": "thinking", "thinking": "think"},
		map[string]any{"type": "text", "text": "answer"},
		map[string]any{"type": "redacted_thinking", "data": "opaque"},
	}}, false)
	body := recorder.Body.String()
	for _, want := range []string{
		"response.created", "response.in_progress", "response.reasoning_summary_part.added",
		"response.reasoning_summary_text.done", "response.content_part.done", "response.completed", "opaque",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("stream missing %q: %s", want, body)
		}
	}
	if got := strings.Count(body, `"encrypted_content":"opaque"`); got != 2 {
		t.Fatalf("opaque reasoning event count = %d, want 2 (item done and completed response): %s", got, body)
	}

	failure := httptest.NewRecorder()
	failingWriter := newStreamWriter(failure, failure, surfaceResponses, "resp_2", "model", "", nil)
	failingWriter.start()
	failingWriter.error("upstream failed")
	if body := failure.Body.String(); !strings.Contains(body, "response.failed") || strings.Contains(body, "event: error") {
		t.Fatalf("failure stream = %s", body)
	}
}

func TestCompatibilityFixturesNormalize(t *testing.T) {
	tests := []struct {
		name string
		path string
		run  func([]byte) error
	}{
		{
			name: "opencode chat", path: "testdata/opencode_chat_image_tool.json",
			run: func(raw []byte) error {
				var request ChatRequest
				if err := json.Unmarshal(raw, &request); err != nil {
					return err
				}
				_, err := request.Normalize()
				return err
			},
		},
		{
			name: "codex responses", path: "testdata/codex_responses_tool_round.json",
			run: func(raw []byte) error {
				var request ResponsesRequest
				if err := json.Unmarshal(raw, &request); err != nil {
					return err
				}
				_, err := request.Normalize()
				return err
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw, err := os.ReadFile(tt.path)
			if err != nil {
				t.Fatal(err)
			}
			if err := tt.run(raw); err != nil {
				t.Fatal(err)
			}
		})
	}
}
