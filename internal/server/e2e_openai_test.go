package server

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func postOpenAI(t *testing.T, url, path, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestOpenAIChatCompletions_TextAndTools(t *testing.T) {
	client := &capturingClient{events: textEvents("hello from Kiro")}
	ts := newE2EServer(t, client)
	defer ts.Close()

	resp := postOpenAI(t, ts.URL, "/v1/chat/completions", `{"model":"claude-sonnet-4-6","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"read_file","description":"read","parameters":{"type":"object"}}}]}`)
	defer resp.Body.Close()
	requireStatus(t, resp, http.StatusOK)
	body := decodeResponse(t, resp)
	choices := body["choices"].([]any)
	message := choices[0].(map[string]any)["message"].(map[string]any)
	if got := message["content"]; got != "hello from Kiro" {
		t.Fatalf("content = %q", got)
	}
	requireCaptured(t, client)
	if got := client.captured.ConversationState.CurrentMessage.UserInputMessage.Content; got != "hello" {
		t.Fatalf("upstream content = %q", got)
	}
	if got := len(client.captured.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext.Tools); got != 1 {
		t.Fatalf("tool count = %d, want 1", got)
	}
}

func TestOpenAIResponses_StreamAndContinuation(t *testing.T) {
	client := &multiResponseClient{responses: [][]any{textEvents("first"), textEvents("second")}}
	ts := newE2EServerWithClient(t, client)
	defer ts.Close()

	first := postOpenAI(t, ts.URL, "/v1/responses", `{"model":"claude-sonnet-4-6","input":"first"}`)
	requireStatus(t, first, http.StatusOK)
	firstBody := decodeResponse(t, first)
	_ = first.Body.Close()
	responseID := firstBody["id"].(string)

	second := postOpenAI(t, ts.URL, "/v1/responses", `{"model":"claude-sonnet-4-6","previous_response_id":"`+responseID+`","input":"second","stream":true}`)
	defer second.Body.Close()
	requireStatus(t, second, http.StatusOK)
	body, err := io.ReadAll(second.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), `"type":"response.created"`) || !strings.Contains(string(body), `"delta":"second"`) || !strings.Contains(string(body), `"type":"response.completed"`) {
		t.Fatalf("unexpected stream: %s", body)
	}
	if len(client.payloads) != 2 {
		t.Fatalf("upstream calls = %d, want 2", len(client.payloads))
	}
	if client.payloads[0].ConversationState.ConversationID != client.payloads[1].ConversationState.ConversationID {
		t.Fatal("previous_response_id did not preserve Kiro conversation")
	}
}

func TestOpenAIResponses_CodexCustomAndNamespaceTools(t *testing.T) {
	toolEvent := mustJSON(map[string]any{
		"name": "apply_patch", "toolUseId": "call_patch", "input": map[string]any{
			"input": "*** Begin Patch\n*** Update File: main.go",
		}, "stop": true,
	})
	client := &multiResponseClient{responses: [][]any{
		{"toolUseEvent", toolEvent},
		textEvents("patch applied"),
	}}
	ts := newE2EServerWithClient(t, client)
	defer ts.Close()

	first := postOpenAI(t, ts.URL, "/v1/responses", `{
		"model":"claude-sonnet-4-6",
		"input":"update the file",
		"tools":[
			{"type":"function","name":"read_file","parameters":{"type":"object"}},
			{"type":"custom","name":"apply_patch","description":"Apply a patch","format":{"type":"text"}},
			{"type":"namespace","name":"mcp__pencil","tools":[{"type":"function","name":"execute","parameters":{"type":"object"}}]},
			{"type":"web_search"}
		]
	}`)
	requireStatus(t, first, http.StatusOK)
	firstBody := decodeResponse(t, first)
	_ = first.Body.Close()

	output := firstBody["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output count = %d, want 1", len(output))
	}
	call := output[0].(map[string]any)
	if call["type"] != "custom_tool_call" || call["name"] != "apply_patch" || call["input"] != "*** Begin Patch\n*** Update File: main.go" {
		t.Fatalf("custom tool call = %#v", call)
	}
	if len(client.payloads) != 1 {
		t.Fatalf("upstream calls = %d, want 1", len(client.payloads))
	}
	tools := client.payloads[0].ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext.Tools
	if len(tools) != 3 {
		t.Fatalf("Kiro tool count = %d, want 3", len(tools))
	}
	if tools[1].ToolSpecification.Name != "apply_patch" || tools[2].ToolSpecification.Name != "mcp__pencil__execute" {
		t.Fatalf("Kiro tool names = %q, %q, %q", tools[0].ToolSpecification.Name, tools[1].ToolSpecification.Name, tools[2].ToolSpecification.Name)
	}

	second := postOpenAI(t, ts.URL, "/v1/responses", `{
		"model":"claude-sonnet-4-6",
		"previous_response_id":"`+firstBody["id"].(string)+`",
		"input":[{"type":"custom_tool_call_output","call_id":"call_patch","output":"Done"}]
	}`)
	requireStatus(t, second, http.StatusOK)
	_ = decodeResponse(t, second)
	_ = second.Body.Close()
	if len(client.payloads) != 2 {
		t.Fatalf("upstream calls = %d, want 2", len(client.payloads))
	}
	if !strings.Contains(historyText(client.payloads[1]), "Done") {
		t.Fatalf("custom tool output missing from Kiro request: %s", historyText(client.payloads[1]))
	}
}
