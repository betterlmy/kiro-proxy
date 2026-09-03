package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/betterlmy/kiro-proxy/internal/anthropic"
)

func normalizeChatTools(tools []ChatTool) ([]anthropic.Tool, error) {
	result := make([]anthropic.Tool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "function" {
			return nil, fmt.Errorf("unsupported tool type %q; only function is supported", tool.Type)
		}
		if tool.Function.Name == "" {
			return nil, fmt.Errorf("function tool name is required")
		}
		result = append(result, anthropic.Tool{Name: tool.Function.Name, Description: tool.Function.Description, InputSchema: tool.Function.Parameters})
	}
	return result, nil
}

func normalizeResponseTools(tools []ResponseTool) ([]anthropic.Tool, error) {
	result := make([]anthropic.Tool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "function" {
			return nil, fmt.Errorf("unsupported tool type %q; only function is supported", tool.Type)
		}
		if tool.Name == "" {
			return nil, fmt.Errorf("function tool name is required")
		}
		result = append(result, anthropic.Tool{Name: tool.Name, Description: tool.Description, InputSchema: tool.Parameters})
	}
	return result, nil
}

func responseInputMessages(input any) ([]anthropic.Message, error) {
	if text, ok := input.(string); ok {
		return []anthropic.Message{{Role: "user", Content: anthropic.MessageContent{Text: text}}}, nil
	}
	items, ok := input.([]any)
	if !ok {
		return nil, fmt.Errorf("input must be a string or an array of input items")
	}
	result := make([]anthropic.Message, 0, len(items))
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("input items must be objects")
		}
		typ, _ := item["type"].(string)
		switch typ {
		case "message", "":
			role, _ := item["role"].(string)
			content, err := contentText(item["content"])
			if err != nil {
				return nil, err
			}
			switch role {
			case "user":
				result = append(result, anthropic.Message{Role: "user", Content: anthropic.MessageContent{Text: content}})
			case "assistant":
				result = append(result, anthropic.Message{Role: "assistant", Content: anthropic.MessageContent{Text: content}})
			case "system", "developer":
				// Responses' system/developer items are treated as a user-side
				// instruction because this API surface carries no separate field.
				result = append(result, anthropic.Message{Role: "user", Content: anthropic.MessageContent{Text: content}})
			default:
				return nil, fmt.Errorf("unsupported input message role %q", role)
			}
		case "function_call_output":
			callID, _ := item["call_id"].(string)
			if callID == "" {
				return nil, fmt.Errorf("function_call_output requires call_id")
			}
			content, err := contentText(item["output"])
			if err != nil {
				return nil, err
			}
			result = append(result, anthropic.Message{Role: "user", Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{{Type: anthropic.BlockTypeToolResult, ToolUseID: callID, Content: anthropic.MessageContent{Text: content}}}}})
		case "function_call":
			callID, _ := item["call_id"].(string)
			name, _ := item["name"].(string)
			arguments, _ := item["arguments"].(string)
			input, err := objectFromJSON(arguments)
			if err != nil {
				return nil, fmt.Errorf("invalid function_call arguments: %w", err)
			}
			result = append(result, anthropic.Message{Role: "assistant", Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{{Type: anthropic.BlockTypeToolUse, ID: callID, Name: name, Input: input}}}})
		default:
			return nil, fmt.Errorf("unsupported input item type %q", typ)
		}
	}
	return result, nil
}

func contentText(content any) (string, error) {
	if content == nil {
		return "", nil
	}
	if text, ok := content.(string); ok {
		return text, nil
	}
	parts, ok := content.([]any)
	if !ok {
		return "", fmt.Errorf("unsupported content format")
	}
	var text []string
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			return "", fmt.Errorf("content parts must be objects")
		}
		typ, _ := part["type"].(string)
		switch typ {
		case "text", "input_text", "output_text":
			value, _ := part["text"].(string)
			text = append(text, value)
		default:
			return "", fmt.Errorf("unsupported content part type %q", typ)
		}
	}
	return strings.Join(text, "\n"), nil
}

func objectFromJSON(raw string) (map[string]any, error) {
	if strings.TrimSpace(raw) == "" {
		return map[string]any{}, nil
	}
	var value map[string]any
	if err := json.Unmarshal([]byte(raw), &value); err != nil {
		return nil, err
	}
	return value, nil
}

func stopSequences(raw any) ([]string, error) {
	switch v := raw.(type) {
	case nil:
		return nil, nil
	case string:
		return []string{v}, nil
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			text, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("stop must contain strings")
			}
			out = append(out, text)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("stop must be a string or an array of strings")
	}
}
