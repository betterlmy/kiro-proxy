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
