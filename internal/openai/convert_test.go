package openai

import (
	"net/http/httptest"
	"strings"
	"testing"

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
