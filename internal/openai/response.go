package openai

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/betterlmy/kiro-proxy/internal/respconv"
	"github.com/google/uuid"
)

func chatResponse(id, model string, source map[string]any) map[string]any {
	content, toolCalls, reasoning := responseParts(source)
	message := map[string]any{"role": "assistant", "content": content}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}
	if reasoning != "" {
		// reasoning_content is widely used by OpenAI-compatible coding clients.
		// The standard request-side control remains reasoning_effort.
		message["reasoning_content"] = reasoning
	}
	finish := "stop"
	if len(toolCalls) > 0 {
		finish = "tool_calls"
	} else if source["stop_reason"] == respconv.StopReasonMaxTokens {
		finish = "length"
	}
	return map[string]any{
		"id": id, "object": "chat.completion", "created": now(), "model": model,
		"choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finish}},
		"usage":   chatUsage(source),
	}
}

func responsesResponse(id, model string, source map[string]any, previousID string, toolKinds map[string]responseToolKind) (map[string]any, error) {
	output, err := responseOutput(source, toolKinds)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"id": id, "object": "response", "created_at": now(), "status": "completed", "completed_at": now(),
		"error": nil, "incomplete_details": nil, "model": model, "output": output,
		"previous_response_id": nullIfEmpty(previousID), "store": false,
		"parallel_tool_calls": true, "tool_choice": "auto", "usage": responseUsage(source),
	}, nil
}

func responseParts(source map[string]any) (string, []any, string) {
	var text, reasoning string
	var tools []any
	blocks, _ := source["content"].([]any)
	for _, raw := range blocks {
		block, _ := raw.(map[string]any)
		switch block["type"] {
		case "text":
			text += stringValue(block["text"])
		case "thinking":
			reasoning += stringValue(block["thinking"])
		case "tool_use":
			args, _ := json.Marshal(block["input"])
			tools = append(tools, map[string]any{
				"id": stringValue(block["id"]), "type": "function",
				"function": map[string]any{"name": stringValue(block["name"]), "arguments": string(args)},
			})
		}
	}
	return text, tools, reasoning
}

func responseOutput(source map[string]any, toolKinds map[string]responseToolKind) ([]any, error) {
	text, tools, reasoning := responseParts(source)
	output := make([]any, 0, len(tools)+2)
	if reasoning != "" {
		output = append(output, map[string]any{"id": "rs_" + idSuffix(), "type": "reasoning", "summary": []any{map[string]any{"type": "summary_text", "text": reasoning}}})
	}
	if text != "" {
		output = append(output, map[string]any{"id": "msg_" + idSuffix(), "type": "message", "role": "assistant", "status": "completed", "content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}}})
	}
	for _, raw := range tools {
		tool := raw.(map[string]any)
		fn := tool["function"].(map[string]any)
		call, err := responseToolCall(stringValue(tool["id"]), stringValue(fn["name"]), stringValue(fn["arguments"]), toolKinds)
		if err != nil {
			return nil, err
		}
		output = append(output, call)
	}
	return output, nil
}

func responseToolCall(callID, name, arguments string, toolKinds map[string]responseToolKind) (map[string]any, error) {
	if toolKinds[name] != responseToolKindCustom {
		return map[string]any{"id": "fc_" + idSuffix(), "type": "function_call", "call_id": callID, "name": name, "arguments": arguments, "status": "completed"}, nil
	}
	input, err := customToolInput(name, arguments)
	if err != nil {
		return nil, err
	}
	return map[string]any{"id": "ctc_" + idSuffix(), "type": "custom_tool_call", "call_id": callID, "name": name, "input": input, "status": "completed"}, nil
}

func customToolInput(name, arguments string) (string, error) {
	input, err := objectFromJSON(arguments)
	if err != nil {
		return "", fmt.Errorf("invalid input for custom tool %q: %w", name, err)
	}
	value, ok := input["input"].(string)
	if !ok {
		return "", fmt.Errorf("custom tool %q must return an input string", name)
	}
	return value, nil
}

func chatUsage(source map[string]any) map[string]any {
	usage, _ := source["usage"].(map[string]any)
	in := intValue(usage["input_tokens"])
	out := intValue(usage["output_tokens"])
	return map[string]any{"prompt_tokens": in, "completion_tokens": out, "total_tokens": in + out}
}

func responseUsage(source map[string]any) map[string]any {
	usage, _ := source["usage"].(map[string]any)
	in := intValue(usage["input_tokens"])
	out := intValue(usage["output_tokens"])
	return map[string]any{"input_tokens": in, "output_tokens": out, "total_tokens": in + out, "input_tokens_details": map[string]any{"cached_tokens": intValue(usage["cache_read_input_tokens"])}, "output_tokens_details": map[string]any{"reasoning_tokens": 0}}
}

func nullIfEmpty(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func stringValue(v any) string { value, _ := v.(string); return value }
func intValue(v any) int       { value, _ := v.(int); return value }
func idSuffix() string         { return uuid.NewString() }

type streamWriter struct {
	w                     http.ResponseWriter
	f                     http.Flusher
	api                   surface
	id, model, previousID string
	sequence              int
	chatRole              bool
	chatToolIndex         int
	responseTextOpen      bool
	responseOutputIndex   int
	toolKinds             map[string]responseToolKind
	failed                bool
}

func newStreamWriter(w http.ResponseWriter, f http.Flusher, api surface, id, model, previousID string, toolKinds map[string]responseToolKind) *streamWriter {
	return &streamWriter{w: w, f: f, api: api, id: id, model: model, previousID: previousID, toolKinds: toolKinds}
}

func (s *streamWriter) start() {
	if s.api == surfaceChat {
		return
	}
	s.event(map[string]any{"type": "response.created", "response": map[string]any{"id": s.id, "object": "response", "created_at": now(), "status": "in_progress", "model": s.model, "output": []any{}, "usage": nil, "previous_response_id": nullIfEmpty(s.previousID)}})
}

func (s *streamWriter) delta(delta respconv.EventDelta) {
	if delta.ThinkingDelta != "" {
		if s.api == surfaceChat {
			s.chat(map[string]any{"reasoning_content": delta.ThinkingDelta})
		} else {
			s.event(map[string]any{"type": "response.reasoning_summary_text.delta", "delta": delta.ThinkingDelta, "output_index": s.responseOutputIndex, "summary_index": 0, "item_id": "rs_" + s.id})
		}
	}
	if delta.TextDelta != "" {
		if s.api == surfaceChat {
			s.chat(map[string]any{"content": delta.TextDelta})
			return
		}
		if !s.responseTextOpen {
			s.responseTextOpen = true
			s.event(map[string]any{"type": "response.output_item.added", "output_index": s.responseOutputIndex, "item": map[string]any{"id": "msg_" + s.id, "type": "message", "role": "assistant", "status": "in_progress", "content": []any{}}})
			s.event(map[string]any{"type": "response.content_part.added", "output_index": s.responseOutputIndex, "content_index": 0, "item_id": "msg_" + s.id, "part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}})
		}
		s.event(map[string]any{"type": "response.output_text.delta", "output_index": s.responseOutputIndex, "content_index": 0, "item_id": "msg_" + s.id, "delta": delta.TextDelta})
	}
	if delta.ToolStop {
		if s.responseTextOpen && s.api == surfaceResponses {
			s.event(map[string]any{"type": "response.output_text.done", "output_index": s.responseOutputIndex, "content_index": 0, "item_id": "msg_" + s.id, "text": ""})
			s.event(map[string]any{"type": "response.output_item.done", "output_index": s.responseOutputIndex, "item": map[string]any{"id": "msg_" + s.id, "type": "message", "role": "assistant", "status": "completed", "content": []any{}}})
			s.responseTextOpen = false
			s.responseOutputIndex++
		}
		if s.api == surfaceChat {
			s.chat(map[string]any{"tool_calls": []any{map[string]any{"index": s.chatToolIndex, "id": delta.ToolUseID, "type": "function", "function": map[string]any{"name": delta.ToolName, "arguments": delta.ToolInput}}}})
			s.chatToolIndex++
		} else {
			call, err := responseToolCall(delta.ToolUseID, delta.ToolName, delta.ToolInput, s.toolKinds)
			if err != nil {
				s.failed = true
				s.error(err.Error())
				return
			}
			if call["type"] == "custom_tool_call" {
				s.writeCustomToolCall(call)
			} else {
				s.writeFunctionToolCall(call)
			}
			s.responseOutputIndex++
		}
	}
}

func (s *streamWriter) finish(source map[string]any, includeUsage bool) {
	if s.api == surfaceChat {
		finish := "stop"
		_, tools, _ := responseParts(source)
		if len(tools) > 0 {
			finish = "tool_calls"
		}
		s.chatFinal(finish, includeUsage, chatUsage(source))
		return
	}
	if s.responseTextOpen {
		text, _, _ := responseParts(source)
		s.event(map[string]any{"type": "response.output_text.done", "output_index": s.responseOutputIndex, "content_index": 0, "item_id": "msg_" + s.id, "text": text})
		s.event(map[string]any{"type": "response.output_item.done", "output_index": s.responseOutputIndex, "item": map[string]any{"id": "msg_" + s.id, "type": "message", "role": "assistant", "status": "completed", "content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}}}})
	}
	if s.failed {
		return
	}
	response, err := responsesResponse(s.id, s.model, source, s.previousID, s.toolKinds)
	if err != nil {
		s.error(err.Error())
		return
	}
	s.event(map[string]any{"type": "response.completed", "response": response})
}

func (s *streamWriter) writeFunctionToolCall(call map[string]any) {
	itemID := "fc_" + stringValue(call["call_id"])
	arguments := stringValue(call["arguments"])
	s.event(map[string]any{"type": "response.output_item.added", "output_index": s.responseOutputIndex, "item": map[string]any{"id": itemID, "type": "function_call", "call_id": call["call_id"], "name": call["name"], "arguments": "", "status": "in_progress"}})
	s.event(map[string]any{"type": "response.function_call_arguments.delta", "output_index": s.responseOutputIndex, "item_id": itemID, "delta": arguments})
	s.event(map[string]any{"type": "response.function_call_arguments.done", "output_index": s.responseOutputIndex, "item_id": itemID, "arguments": arguments})
	s.event(map[string]any{"type": "response.output_item.done", "output_index": s.responseOutputIndex, "item": map[string]any{"id": itemID, "type": "function_call", "call_id": call["call_id"], "name": call["name"], "arguments": arguments, "status": "completed"}})
}

func (s *streamWriter) writeCustomToolCall(call map[string]any) {
	itemID := "ctc_" + stringValue(call["call_id"])
	input := stringValue(call["input"])
	s.event(map[string]any{"type": "response.output_item.added", "output_index": s.responseOutputIndex, "item": map[string]any{"id": itemID, "type": "custom_tool_call", "call_id": call["call_id"], "name": call["name"], "input": "", "status": "in_progress"}})
	s.event(map[string]any{"type": "response.custom_tool_call_input.delta", "output_index": s.responseOutputIndex, "item_id": itemID, "delta": input})
	s.event(map[string]any{"type": "response.custom_tool_call_input.done", "output_index": s.responseOutputIndex, "item_id": itemID, "input": input})
	s.event(map[string]any{"type": "response.output_item.done", "output_index": s.responseOutputIndex, "item": map[string]any{"id": itemID, "type": "custom_tool_call", "call_id": call["call_id"], "name": call["name"], "input": input, "status": "completed"}})
}

func (s *streamWriter) error(message string) {
	if s.api == surfaceChat {
		s.write(map[string]any{"error": map[string]any{"message": message, "type": "server_error"}})
		return
	}
	s.event(map[string]any{"type": "error", "error": map[string]any{"message": message, "type": "server_error"}})
}

func (s *streamWriter) chat(delta map[string]any) {
	if !s.chatRole {
		delta["role"] = "assistant"
		s.chatRole = true
	}
	s.write(map[string]any{"id": s.id, "object": "chat.completion.chunk", "created": now(), "model": s.model, "choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": nil}}})
}
func (s *streamWriter) chatFinal(reason string, includeUsage bool, usage map[string]any) {
	value := map[string]any{"id": s.id, "object": "chat.completion.chunk", "created": now(), "model": s.model, "choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": reason}}}
	if includeUsage {
		value["usage"] = usage
	}
	s.write(value)
	_, _ = s.w.Write([]byte("data: [DONE]\n\n"))
	if s.f != nil {
		s.f.Flush()
	}
}
func (s *streamWriter) event(value map[string]any) {
	s.sequence++
	value["sequence_number"] = s.sequence
	eventType, _ := value["type"].(string)
	body, _ := json.Marshal(value)
	_, _ = s.w.Write([]byte("event: " + eventType + "\n"))
	_, _ = s.w.Write([]byte("data: "))
	_, _ = s.w.Write(body)
	_, _ = s.w.Write([]byte("\n\n"))
	if s.f != nil {
		s.f.Flush()
	}
}
func (s *streamWriter) write(value any) {
	body, _ := json.Marshal(value)
	_, _ = s.w.Write([]byte("data: "))
	_, _ = s.w.Write(body)
	_, _ = s.w.Write([]byte("\n\n"))
	if s.f != nil {
		s.f.Flush()
	}
}
