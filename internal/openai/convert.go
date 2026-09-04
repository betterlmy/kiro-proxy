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

type normalizedResponseInput struct {
	Messages        []anthropic.Message
	AdditionalTools []ResponseTool
}

func responseInputMessages(input any) ([]anthropic.Message, error) {
	normalized, err := normalizeResponseInput(input)
	return normalized.Messages, err
}

func normalizeResponseInput(input any) (normalizedResponseInput, error) {
	if text, ok := input.(string); ok {
		return normalizedResponseInput{Messages: []anthropic.Message{{Role: "user", Content: anthropic.MessageContent{Text: text}}}}, nil
	}
	items, ok := input.([]any)
	if !ok {
		return normalizedResponseInput{}, fmt.Errorf("input must be a string or an array of input items")
	}
	result := normalizedResponseInput{Messages: make([]anthropic.Message, 0, len(items))}
	toolIDs := newToolCallIDMap()
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			return normalizedResponseInput{}, fmt.Errorf("input items must be objects")
		}
		additional, err := responseAdditionalTools(item)
		if err != nil {
			return normalizedResponseInput{}, err
		}
		result.AdditionalTools = append(result.AdditionalTools, additional...)
		typ, _ := item["type"].(string)
		switch typ {
		case "message", "":
			role, _ := item["role"].(string)
			content, err := openAIContent(item["content"])
			if err != nil {
				return normalizedResponseInput{}, err
			}
			switch role {
			case "user", "assistant":
				result.Messages = appendNormalizedMessage(result.Messages, anthropic.Message{Role: role, Content: content})
			case "system", "developer":
				text, err := textOnlyContent(content)
				if err != nil {
					return normalizedResponseInput{}, fmt.Errorf("%s input message: %w", role, err)
				}
				result.Messages = appendNormalizedMessage(result.Messages, anthropic.Message{Role: "user", Content: anthropic.MessageContent{Text: text}})
			default:
				return normalizedResponseInput{}, fmt.Errorf("unsupported input message role %q", role)
			}
		case "function_call_output", "custom_tool_call_output":
			rawCallID, _ := item["call_id"].(string)
			callID, err := toolIDs.Resolve(rawCallID)
			if err != nil {
				return normalizedResponseInput{}, fmt.Errorf("%s: %w", typ, err)
			}
			if callID == "" {
				return normalizedResponseInput{}, fmt.Errorf("%s requires call_id", typ)
			}
			content, err := openAIContent(item["output"])
			if err != nil {
				return normalizedResponseInput{}, err
			}
			result.Messages = appendToolResultMessage(result.Messages, callID, content)
		case "function_call":
			rawCallID, _ := item["call_id"].(string)
			callID, err := toolIDs.Register(rawCallID)
			if err != nil {
				return normalizedResponseInput{}, fmt.Errorf("function_call: %w", err)
			}
			name, _ := item["name"].(string)
			if callID == "" || name == "" {
				return normalizedResponseInput{}, fmt.Errorf("function_call requires call_id and name")
			}
			arguments, _ := item["arguments"].(string)
			toolInput, err := objectFromJSON(arguments)
			if err != nil {
				return normalizedResponseInput{}, fmt.Errorf("invalid function_call arguments: %w", err)
			}
			result.Messages = appendNormalizedMessage(result.Messages, anthropic.Message{Role: "assistant", Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{{Type: anthropic.BlockTypeToolUse, ID: callID, Name: name, Input: toolInput}}}})
		case "custom_tool_call":
			rawCallID, _ := item["call_id"].(string)
			callID, err := toolIDs.Register(rawCallID)
			if err != nil {
				return normalizedResponseInput{}, fmt.Errorf("custom_tool_call: %w", err)
			}
			name, _ := item["name"].(string)
			toolInput, _ := item["input"].(string)
			if callID == "" || name == "" {
				return normalizedResponseInput{}, fmt.Errorf("custom_tool_call requires call_id and name")
			}
			result.Messages = appendNormalizedMessage(result.Messages, anthropic.Message{Role: "assistant", Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{{Type: anthropic.BlockTypeToolUse, ID: callID, Name: name, Input: map[string]any{"input": toolInput}}}}})
		case "reasoning":
			encrypted, _ := item["encrypted_content"].(string)
			if encrypted != "" {
				result.Messages = appendNormalizedMessage(result.Messages, anthropic.Message{Role: "assistant", Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{{Type: anthropic.BlockTypeRedactedThinking, Data: encrypted}}}})
			}
		default:
			return normalizedResponseInput{}, fmt.Errorf("unsupported input item type %q", typ)
		}
	}
	return result, nil
}

func responseAdditionalTools(item map[string]any) ([]ResponseTool, error) {
	raw, ok := item["additional_tools"]
	if !ok || raw == nil {
		return nil, nil
	}
	encoded, err := json.Marshal(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid additional_tools: %w", err)
	}
	var tools []ResponseTool
	if err := json.Unmarshal(encoded, &tools); err != nil {
		return nil, fmt.Errorf("additional_tools must be an array of tools: %w", err)
	}
	return tools, nil
}

// appendNormalizedMessage preserves an OpenAI tool-call batch as one Anthropic
// turn for both Chat Completions and Responses. Kiro requires all parallel calls
// in an assistant turn and their results in the following user turn.
func appendNormalizedMessage(messages []anthropic.Message, next anthropic.Message) []anthropic.Message {
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
