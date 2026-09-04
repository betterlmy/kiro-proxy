package openai

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/betterlmy/kiro-proxy/internal/anthropic"
)

const maxOpenAIImageBytes = 3 << 20

// openAIContent converts the shared Chat Completions and Responses content
// subset into the proxy's canonical content representation. Kiro accepts only
// inline images, so remote URLs and file IDs are rejected at this boundary.
func openAIContent(content any) (anthropic.MessageContent, error) {
	if content == nil {
		return anthropic.MessageContent{Text: ""}, nil
	}
	if text, ok := content.(string); ok {
		return anthropic.MessageContent{Text: text}, nil
	}
	parts, ok := content.([]any)
	if !ok {
		return anthropic.MessageContent{}, fmt.Errorf("unsupported content format")
	}
	blocks := make([]anthropic.ContentBlock, 0, len(parts))
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			return anthropic.MessageContent{}, fmt.Errorf("content parts must be objects")
		}
		typ, _ := part["type"].(string)
		switch typ {
		case "text", "input_text", "output_text":
			text, _ := part["text"].(string)
			blocks = append(blocks, anthropic.ContentBlock{Type: anthropic.BlockTypeText, Text: text})
		case "refusal":
			text, _ := part["refusal"].(string)
			blocks = append(blocks, anthropic.ContentBlock{Type: anthropic.BlockTypeText, Text: text})
		case "image_url", "input_image":
			image, err := openAIImageBlock(part)
			if err != nil {
				return anthropic.MessageContent{}, err
			}
			blocks = append(blocks, image)
		case "input_file":
			image, err := openAIFileImageBlock(part)
			if err != nil {
				return anthropic.MessageContent{}, err
			}
			blocks = append(blocks, image)
		default:
			return anthropic.MessageContent{}, fmt.Errorf("unsupported content part type %q", typ)
		}
	}
	return anthropic.MessageContent{Blocks: blocks}, nil
}

func textOnlyContent(content anthropic.MessageContent) (string, error) {
	if content.IsString() {
		return content.Text, nil
	}
	text := make([]string, 0, len(content.Blocks))
	for _, block := range content.Blocks {
		if block.Type != anthropic.BlockTypeText {
			return "", fmt.Errorf("content may contain only text parts")
		}
		text = append(text, block.Text)
	}
	return strings.Join(text, "\n"), nil
}

func openAIImageBlock(part map[string]any) (anthropic.ContentBlock, error) {
	var rawURL string
	if value, ok := part["image_url"].(string); ok {
		rawURL = value
	} else if value, ok := part["image_url"].(map[string]any); ok {
		rawURL, _ = value["url"].(string)
	}
	if rawURL == "" {
		typ, _ := part["type"].(string)
		return anthropic.ContentBlock{}, fmt.Errorf("%s requires an inline image_url", typ)
	}
	return dataURLImageBlock(rawURL)
}

func openAIFileImageBlock(part map[string]any) (anthropic.ContentBlock, error) {
	raw, _ := part["file_data"].(string)
	if raw == "" {
		return anthropic.ContentBlock{}, fmt.Errorf("input_file supports only inline image file_data; file_id and remote files are not supported")
	}
	return dataURLImageBlock(raw)
}

func dataURLImageBlock(raw string) (anthropic.ContentBlock, error) {
	comma := strings.IndexByte(raw, ',')
	if comma <= 0 {
		return anthropic.ContentBlock{}, fmt.Errorf("image_url must be a base64 data URL; remote image URLs are not supported")
	}
	meta := strings.ToLower(raw[:comma])
	if !strings.HasPrefix(meta, "data:image/") || !strings.HasSuffix(meta, ";base64") {
		return anthropic.ContentBlock{}, fmt.Errorf("image_url must use a supported base64 image media type")
	}
	mediaType := strings.TrimSuffix(strings.TrimPrefix(meta, "data:"), ";base64")
	switch mediaType {
	case "image/png", "image/jpeg", "image/gif", "image/webp":
	default:
		return anthropic.ContentBlock{}, fmt.Errorf("unsupported image media type %q", mediaType)
	}
	data := raw[comma+1:]
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return anthropic.ContentBlock{}, fmt.Errorf("invalid base64 image data: %w", err)
	}
	if len(decoded) > maxOpenAIImageBytes {
		return anthropic.ContentBlock{}, fmt.Errorf("image exceeds %d byte limit", maxOpenAIImageBytes)
	}
	return anthropic.ContentBlock{Type: anthropic.BlockTypeImage, Source: &anthropic.ImageSource{Type: "base64", MediaType: mediaType, Data: data}}, nil
}

// appendToolResultMessage keeps only the final result for one call ID in an
// adjacent user turn. This repairs clients that replay partial tool outputs.
func appendToolResultMessage(messages []anthropic.Message, callID string, content anthropic.MessageContent) []anthropic.Message {
	block := anthropic.ContentBlock{Type: anthropic.BlockTypeToolResult, ToolUseID: callID, Content: content}
	if len(messages) == 0 || messages[len(messages)-1].Role != "user" {
		return append(messages, anthropic.Message{Role: "user", Content: anthropic.MessageContent{Blocks: []anthropic.ContentBlock{block}}})
	}
	previous := &messages[len(messages)-1]
	blocks := contentBlocks(previous.Content)
	for i := len(blocks) - 1; i >= 0; i-- {
		if blocks[i].Type == anthropic.BlockTypeToolResult && blocks[i].ToolUseID == callID {
			blocks = append(blocks[:i], blocks[i+1:]...)
		}
	}
	previous.Content = anthropic.MessageContent{Blocks: append(blocks, block)}
	return messages
}

type toolCallIDMap struct {
	byOriginal map[string]string
}

func newToolCallIDMap() *toolCallIDMap {
	return &toolCallIDMap{byOriginal: make(map[string]string)}
}

func (m *toolCallIDMap) Register(id string) (string, error) {
	if id == "" {
		return "", nil
	}
	if _, exists := m.byOriginal[id]; exists {
		return "", fmt.Errorf("duplicate tool call id %q", id)
	}
	canonical := canonicalToolCallID(id)
	for original, existing := range m.byOriginal {
		if original != id && existing == canonical {
			return "", fmt.Errorf("tool call id %q collides with %q", id, original)
		}
	}
	m.byOriginal[id] = canonical
	return canonical, nil
}

func (m *toolCallIDMap) Resolve(id string) (string, error) {
	if id == "" {
		return "", nil
	}
	if canonical, ok := m.byOriginal[id]; ok {
		return canonical, nil
	}
	// Responses continuations can contain only the tool result. In that form,
	// the matching tool call already exists in Kiro's conversation state.
	return id, nil
}

func canonicalToolCallID(id string) string {
	if len(id) <= 64 && isSafeToolCallID(id) {
		return id
	}
	var b strings.Builder
	for _, r := range id {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	prefix := b.String()
	if len(prefix) > 50 {
		prefix = prefix[:50]
	}
	if prefix == "" {
		prefix = "call"
	}
	hash := sha256.Sum256([]byte(id))
	return fmt.Sprintf("%s_%x", prefix, hash[:6])
}

func isSafeToolCallID(id string) bool {
	if id == "" {
		return false
	}
	for _, r := range id {
		if (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && (r < '0' || r > '9') && r != '_' && r != '-' {
			return false
		}
	}
	return true
}

func normalizeToolChoice(raw any) (any, error) {
	if raw == nil {
		return "auto", nil
	}
	choice, ok := raw.(string)
	if !ok {
		return nil, fmt.Errorf("specific tool_choice is not supported by Kiro")
	}
	switch choice {
	case "auto", "none":
		return choice, nil
	case "required":
		return nil, fmt.Errorf("tool_choice %q cannot be enforced by Kiro", choice)
	default:
		return nil, fmt.Errorf("unsupported tool_choice %q", choice)
	}
}
