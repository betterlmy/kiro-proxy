package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/betterlmy/kiro-proxy/internal/anthropic"
	"github.com/betterlmy/kiro-proxy/internal/auth"
	"github.com/betterlmy/kiro-proxy/internal/kiroclient"
	"github.com/betterlmy/kiro-proxy/internal/kiroproto"
	"github.com/betterlmy/kiro-proxy/internal/models"
	"github.com/betterlmy/kiro-proxy/internal/reqconv"
	"github.com/betterlmy/kiro-proxy/internal/respconv"
	"github.com/google/uuid"
)

// TokenGetter loads upstream credentials for a request.
type TokenGetter interface {
	GetToken(context.Context) (*auth.Credentials, error)
}

// Service serves OpenAI-compatible Chat Completions and Responses requests.
// Response continuation state is intentionally process-local: the Kiro session
// ID is mapped from the opaque response ID only for the server lifetime.
type Service struct {
	auth   TokenGetter
	client kiroclient.Client

	mu            sync.Mutex
	conversations map[string]string
}

func New(auth TokenGetter, client kiroclient.Client) *Service {
	return &Service{auth: auth, client: client, conversations: make(map[string]string)}
}

func (s *Service) HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var input ChatRequest
	if err := decodeRequest(w, r, &input); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	req, err := input.Normalize()
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	s.execute(w, r, req, surfaceChat)
}

func (s *Service) HandleResponses(w http.ResponseWriter, r *http.Request) {
	var input ResponsesRequest
	if err := decodeRequest(w, r, &input); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	req, err := input.Normalize()
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	s.execute(w, r, req, surfaceResponses)
}

type surface uint8

const (
	surfaceChat surface = iota
	surfaceResponses
)

func (s *Service) execute(w http.ResponseWriter, r *http.Request, input Request, api surface) {
	creds, err := s.auth.GetToken(r.Context())
	if err != nil {
		writeError(w, http.StatusUnauthorized, "authentication_error", "authentication failed")
		return
	}
	responseID := "resp_" + uuid.NewString()
	conversationID, ok := s.conversation(input.PreviousResponseID)
	if input.PreviousResponseID != "" && !ok {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "unknown previous_response_id; resend the complete conversation after restarting kiro-proxy")
		return
	}
	if conversationID == "" {
		conversationID = "conv_" + uuid.NewString()
	}

	kiroModel, _, window, responseModel := models.Resolve(input.Model, false)
	effort := models.ResolveEffort(kiroModel, input.ReasoningEffort)
	anthReq := &anthropic.Request{
		Model:         input.Model,
		Messages:      input.Messages,
		System:        anthropic.SystemPrompt{Text: input.System},
		Tools:         input.Tools,
		MaxTokens:     input.MaxTokens,
		StopSequences: input.StopSequences,
		Stream:        input.Stream,
	}
	if input.ReasoningEffort != "" {
		anthReq.OutputConfig = &anthropic.OutputConfig{Effort: input.ReasoningEffort}
	}
	payload, names, err := reqconv.BuildPayload(anthReq, reqconv.BuildOptions{
		ProfileARN: creds.ProfileARN, ModelID: kiroModel, ConversationID: conversationID, Effort: effort,
	})
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	upstream, err := s.client.GenerateAssistantResponse(r.Context(), creds.AccessToken, payload, creds.Region)
	if err != nil {
		writeError(w, http.StatusBadGateway, "server_error", "upstream API error")
		return
	}
	defer upstream.Body.Close()

	acc := respconv.NewNonStreamingAccumulator(window, input.StopSequences, input.MaxTokens, upstream.PromptTokens)
	acc.SetToolNameMap(names.ReverseMap())
	if input.Stream {
		s.stream(w, r.Context(), upstream.Body, acc, input, api, responseID, responseModel, conversationID)
		return
	}
	if err := consume(r.Context(), upstream.Body, acc, nil); err != nil {
		writeError(w, http.StatusBadGateway, "server_error", err.Error())
		return
	}
	response, _ := acc.BuildResponse(responseModel)
	s.remember(responseID, conversationID)
	w.Header().Set("Content-Type", "application/json")
	if api == surfaceChat {
		writeJSON(w, chatResponse(responseID, responseModel, response))
	} else {
		body, err := responsesResponse(responseID, responseModel, response, input.PreviousResponseID, input.ResponseToolKinds)
		if err != nil {
			writeError(w, http.StatusBadGateway, "server_error", err.Error())
			return
		}
		writeJSON(w, body)
	}
}

func (s *Service) stream(w http.ResponseWriter, ctx context.Context, body io.Reader, acc *respconv.NonStreamingAccumulator, input Request, api surface, responseID, model, conversationID string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, _ := w.(http.Flusher)
	writer := newStreamWriter(w, flusher, api, responseID, model, input.PreviousResponseID, input.ResponseToolKinds)
	writer.start()
	err := consume(ctx, body, acc, func(delta respconv.EventDelta) {
		writer.delta(delta)
	})
	if err != nil {
		writer.error(err.Error())
		return
	}
	response, _ := acc.BuildResponse(model)
	s.remember(responseID, conversationID)
	writer.finish(response, input.IncludeUsage)
}

func consume(ctx context.Context, body io.Reader, acc *respconv.NonStreamingAccumulator, onDelta func(respconv.EventDelta)) error {
	var upstreamErr string
	err := kiroproto.ParseStream(ctx, body, func(event kiroproto.Event) bool {
		delta := acc.ProcessEvent(event)
		if delta.IsError {
			upstreamErr = delta.ErrorMessage
			return true
		}
		if onDelta != nil {
			onDelta(delta)
		}
		return false
	})
	if err != nil {
		return fmt.Errorf("upstream stream error")
	}
	if upstreamErr != "" {
		return fmt.Errorf("upstream API error: %s", upstreamErr)
	}
	return nil
}

func (s *Service) conversation(responseID string) (string, bool) {
	if responseID == "" {
		return "", true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	id, ok := s.conversations[responseID]
	return id, ok
}

func (s *Service) remember(responseID, conversationID string) {
	s.mu.Lock()
	s.conversations[responseID] = conversationID
	s.mu.Unlock()
}

func decodeRequest(w http.ResponseWriter, r *http.Request, out any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 4<<20)
	defer r.Body.Close()
	return json.NewDecoder(r.Body).Decode(out)
}

func writeJSON(w http.ResponseWriter, body any) {
	_ = json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, status int, typ, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	writeJSON(w, map[string]any{"error": map[string]any{"message": message, "type": typ}})
}

func now() int64 { return time.Now().Unix() }
