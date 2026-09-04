// Package openai implements the OpenAI-compatible HTTP surfaces exposed by
// kiro-proxy. It deliberately models only the text and function-tool subset
// that Kiro can execute; hosted OpenAI tools are rejected at validation time.
package openai

import (
	"fmt"
	"strings"

	"github.com/betterlmy/kiro-proxy/internal/anthropic"
)

const defaultMaxOutputTokens = 4096

// ChatRequest is the supported subset of POST /v1/chat/completions.
type ChatRequest struct {
	Model               string         `json:"model"`
	Messages            []ChatMessage  `json:"messages"`
	Tools               []ChatTool     `json:"tools,omitempty"`
	ToolChoice          any            `json:"tool_choice,omitempty"`
	MaxCompletionTokens int            `json:"max_completion_tokens,omitempty"`
	MaxTokens           int            `json:"max_tokens,omitempty"`
	Stop                any            `json:"stop,omitempty"`
	Stream              bool           `json:"stream,omitempty"`
	StreamOptions       *StreamOptions `json:"stream_options,omitempty"`
	ReasoningEffort     string         `json:"reasoning_effort,omitempty"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type ChatMessage struct {
	Role      string `json:"role"`
	Content   any    `json:"content"`
	ToolCalls []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

type ChatTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		Parameters  map[string]any `json:"parameters,omitempty"`
	} `json:"function"`
}

// ResponsesRequest is the supported subset of POST /v1/responses.
type ResponsesRequest struct {
	Model              string         `json:"model"`
	Input              any            `json:"input"`
	Instructions       string         `json:"instructions,omitempty"`
	Tools              []ResponseTool `json:"tools,omitempty"`
	MaxOutputTokens    int            `json:"max_output_tokens,omitempty"`
	Stream             bool           `json:"stream,omitempty"`
	PreviousResponseID string         `json:"previous_response_id,omitempty"`
	Reasoning          *struct {
		Effort string `json:"effort,omitempty"`
	} `json:"reasoning,omitempty"`
}

type ResponseTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Tools       []ResponseTool `json:"tools,omitempty"`
}

type responseToolKind string

const (
	responseToolKindFunction responseToolKind = "function"
	responseToolKindCustom   responseToolKind = "custom"
)

// Request is the normalized input shared by both OpenAI API surfaces.
type Request struct {
	Model              string
	Messages           []anthropic.Message
	System             string
	Tools              []anthropic.Tool
	MaxTokens          int
	StopSequences      []string
	Stream             bool
	ReasoningEffort    string
	PreviousResponseID string
	IncludeUsage       bool
	ResponseToolKinds  map[string]responseToolKind
}

func (r ChatRequest) Normalize() (Request, error) {
	if r.Model == "" {
		return Request{}, fmt.Errorf("model is required")
	}
	if len(r.Messages) == 0 {
		return Request{}, fmt.Errorf("messages must not be empty")
	}
	result := Request{Model: r.Model, Stream: r.Stream, ReasoningEffort: r.ReasoningEffort}
	result.MaxTokens = r.MaxCompletionTokens
	if result.MaxTokens == 0 {
		result.MaxTokens = r.MaxTokens
	}
	if result.MaxTokens == 0 {
		result.MaxTokens = defaultMaxOutputTokens
	}
	if r.StreamOptions != nil {
		result.IncludeUsage = r.StreamOptions.IncludeUsage
	}
	var system []string
	for _, message := range r.Messages {
		text, err := contentText(message.Content)
		if err != nil {
			return Request{}, err
		}
		switch message.Role {
		case "system", "developer":
			if text != "" {
				system = append(system, text)
			}
		case "user":
			result.Messages = appendNormalizedMessage(result.Messages, anthropic.Message{Role: "user", Content: anthropic.MessageContent{Text: text}})
		case "assistant":
			blocks := make([]anthropic.ContentBlock, 0, len(message.ToolCalls)+1)
			if text != "" {
				blocks = append(blocks, anthropic.ContentBlock{Type: anthropic.BlockTypeText, Text: text})
			}
			for _, call := range message.ToolCalls {
				if call.Type != "" && call.Type != "function" {
					return Request{}, fmt.Errorf("unsupported assistant tool call type %q", call.Type)
				}
				if call.ID == "" {
					return Request{}, fmt.Errorf("assistant tool call requires id")
				}
				if call.Function.Name == "" {
					return Request{}, fmt.Errorf("assistant tool call requires function name")
				}
				input, err := objectFromJSON(call.Function.Arguments)
				if err != nil {
					return Request{}, fmt.Errorf("invalid tool call arguments: %w", err)
				}
				blocks = append(blocks, anthropic.ContentBlock{Type: anthropic.BlockTypeToolUse, ID: call.ID, Name: call.Function.Name, Input: input})
			}
			if len(blocks) == 0 {
				result.Messages = appendNormalizedMessage(result.Messages, anthropic.Message{Role: "assistant", Content: anthropic.MessageContent{Text: ""}})
			} else {
				result.Messages = appendNormalizedMessage(result.Messages, anthropic.Message{Role: "assistant", Content: anthropic.MessageContent{Blocks: blocks}})
			}
		case "tool", "function":
			if message.ToolCallID == "" {
				return Request{}, fmt.Errorf("tool messages require tool_call_id")
			}
			result.Messages = appendNormalizedMessage(result.Messages, anthropic.Message{Role: "user", Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{{Type: anthropic.BlockTypeToolResult, ToolUseID: message.ToolCallID, Content: anthropic.MessageContent{Text: text}}}}})
		default:
			return Request{}, fmt.Errorf("unsupported message role %q", message.Role)
		}
	}
	result.System = strings.Join(system, "\n\n")
	if len(result.Messages) == 0 {
		return Request{}, fmt.Errorf("messages must contain a user, assistant, or tool message")
	}
	var err error
	result.Tools, err = normalizeChatTools(r.Tools)
	if err != nil {
		return Request{}, err
	}
	result.StopSequences, err = stopSequences(r.Stop)
	return result, err
}

func (r ResponsesRequest) Normalize() (Request, error) {
	if r.Model == "" {
		return Request{}, fmt.Errorf("model is required")
	}
	result := Request{Model: r.Model, System: r.Instructions, Stream: r.Stream, PreviousResponseID: r.PreviousResponseID, MaxTokens: r.MaxOutputTokens}
	if result.MaxTokens == 0 {
		result.MaxTokens = defaultMaxOutputTokens
	}
	if r.Reasoning != nil {
		result.ReasoningEffort = r.Reasoning.Effort
	}
	var err error
	result.Messages, err = responseInputMessages(r.Input)
	if err != nil {
		return Request{}, err
	}
	if len(result.Messages) == 0 {
		return Request{}, fmt.Errorf("input must not be empty")
	}
	result.Tools, result.ResponseToolKinds, err = normalizeResponseTools(r.Tools)
	return result, err
}
