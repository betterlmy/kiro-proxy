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

func normalizeResponseTools(tools []ResponseTool) ([]anthropic.Tool, map[string]responseToolKind, error) {
	result := make([]anthropic.Tool, 0, len(tools))
	kinds := make(map[string]responseToolKind)
	if err := appendResponseTools(&result, kinds, tools, ""); err != nil {
		return nil, nil, err
	}
	return result, kinds, nil
}

func appendResponseTools(result *[]anthropic.Tool, kinds map[string]responseToolKind, tools []ResponseTool, namespace string) error {
	for _, tool := range tools {
		switch tool.Type {
		case "function":
			name, err := responseToolName(namespace, tool.Name, "function")
			if err != nil {
				return err
			}
			if _, exists := kinds[name]; exists {
				return fmt.Errorf("duplicate response tool name %q", name)
			}
			*result = append(*result, anthropic.Tool{Name: name, Description: tool.Description, InputSchema: tool.Parameters})
			kinds[name] = responseToolKindFunction
		case "custom":
			name, err := responseToolName(namespace, tool.Name, "custom")
			if err != nil {
				return err
			}
			if _, exists := kinds[name]; exists {
				return fmt.Errorf("duplicate response tool name %q", name)
			}
			description := tool.Description
			if description != "" {
				description += "\n\n"
			}
			description += "请将完整的原始工具输入放入 input 字符串中。"
			*result = append(*result, anthropic.Tool{Name: name, Description: description, InputSchema: customToolInputSchema()})
			kinds[name] = responseToolKindCustom
		case "namespace":
			name, err := responseToolName(namespace, tool.Name, "namespace")
			if err != nil {
				return err
			}
			if len(tool.Tools) == 0 {
				return fmt.Errorf("namespace tool %q must contain tools", name)
			}
			if err := appendResponseTools(result, kinds, tool.Tools, name); err != nil {
				return err
			}
		case "web_search":
			// Kiro 没有兼容的托管网页搜索实现，省略该工具以避免模型选择无法执行的能力。
		default:
			return fmt.Errorf("unsupported tool type %q", tool.Type)
		}
	}
	return nil
}

func responseToolName(namespace, name, toolType string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("%s tool name is required", toolType)
	}
	if namespace == "" {
		return name, nil
	}
	return namespace + "__" + name, nil
}

func customToolInputSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"input"},
		"properties": map[string]any{
			"input": map[string]any{
				"type":        "string",
				"description": "custom 工具的完整原始输入。",
			},
		},
	}
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
				result = appendResponseMessage(result, anthropic.Message{Role: "user", Content: anthropic.MessageContent{Text: content}})
			case "assistant":
				result = appendResponseMessage(result, anthropic.Message{Role: "assistant", Content: anthropic.MessageContent{Text: content}})
			case "system", "developer":
				// Responses' system/developer items are treated as a user-side
				// instruction because this API surface carries no separate field.
				result = appendResponseMessage(result, anthropic.Message{Role: "user", Content: anthropic.MessageContent{Text: content}})
			default:
				return nil, fmt.Errorf("unsupported input message role %q", role)
			}
		case "function_call_output", "custom_tool_call_output":
			callID, _ := item["call_id"].(string)
			if callID == "" {
				return nil, fmt.Errorf("%s requires call_id", typ)
			}
			content, err := contentText(item["output"])
			if err != nil {
				return nil, err
			}
			result = appendResponseMessage(result, anthropic.Message{Role: "user", Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{{Type: anthropic.BlockTypeToolResult, ToolUseID: callID, Content: anthropic.MessageContent{Text: content}}}}})
		case "function_call":
			callID, _ := item["call_id"].(string)
			name, _ := item["name"].(string)
			arguments, _ := item["arguments"].(string)
			input, err := objectFromJSON(arguments)
			if err != nil {
				return nil, fmt.Errorf("invalid function_call arguments: %w", err)
			}
			result = appendResponseMessage(result, anthropic.Message{Role: "assistant", Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{{Type: anthropic.BlockTypeToolUse, ID: callID, Name: name, Input: input}}}})
		case "custom_tool_call":
			callID, _ := item["call_id"].(string)
			name, _ := item["name"].(string)
			input, _ := item["input"].(string)
			if callID == "" || name == "" {
				return nil, fmt.Errorf("custom_tool_call requires call_id and name")
			}
			result = appendResponseMessage(result, anthropic.Message{Role: "assistant", Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{{Type: anthropic.BlockTypeToolUse, ID: callID, Name: name, Input: map[string]any{"input": input}}}}})
		default:
			return nil, fmt.Errorf("unsupported input item type %q", typ)
		}
	}
	return result, nil
}

// appendResponseMessage preserves an OpenAI Responses tool-call batch as one
// Anthropic message. Kiro requires all parallel calls in an assistant turn and
// their results in the following user turn, rather than alternating one call
// and one result at a time.
func appendResponseMessage(messages []anthropic.Message, next anthropic.Message) []anthropic.Message {
	if len(messages) == 0 || messages[len(messages)-1].Role != next.Role {
		return append(messages, next)
	}
	previous := &messages[len(messages)-1]
	previous.Content = anthropic.MessageContent{Blocks: append(contentBlocks(previous.Content), contentBlocks(next.Content)...)}
	return messages
}

func contentBlocks(content anthropic.MessageContent) []anthropic.ContentBlock {
	if content.IsString() {
		return []anthropic.ContentBlock{{Type: anthropic.BlockTypeText, Text: content.Text}}
	}
	return content.Blocks
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
