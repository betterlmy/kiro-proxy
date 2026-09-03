package openai

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/betterlmy/kiro-proxy/internal/anthropic"
	"github.com/betterlmy/kiro-proxy/internal/kiroclient"
	"github.com/betterlmy/kiro-proxy/internal/logging"
)

func TestLogUpstreamError_RedactsBody(t *testing.T) {
	var buf bytes.Buffer
	old := slog.Default()
	slog.SetDefault(slog.New(logging.NewOTelHandler(&buf, slog.LevelDebug)))
	t.Cleanup(func() { slog.SetDefault(old) })

	logUpstreamError(context.Background(), &kiroclient.UpstreamError{
		Status:    400,
		Exception: "ValidationException",
		Body:      `{"message":"tool schema: must-not-log"}`,
	}, Request{
		PreviousResponseID: "resp_previous",
		Messages: []anthropic.Message{{
			Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{{Type: anthropic.BlockTypeToolResult}}},
		}},
		Tools:     []anthropic.Tool{{}},
		Stream:    true,
		MaxTokens: 16,
	}, "high")

	got := buf.String()
	for _, want := range []string{
		`"error_kind":"upstream_response"`,
		`"upstream_status":400`,
		`"upstream_exception":"ValidationException"`,
		`"continuation":true`,
		`"tool_result_count":1`,
		`"reasoning_effort":"high"`,
		`"validation_category":"tool_schema"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("log missing %s: %s", want, got)
		}
	}
	if strings.Contains(got, "must-not-log") {
		t.Fatalf("log contains upstream response body: %s", got)
	}
}
