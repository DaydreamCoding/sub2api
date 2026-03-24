package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// CursorGatewayService handles request/response transformation between
// OpenAI-compatible API format and Cursor's ConnectRPC BiDi streaming protocol.
type CursorGatewayService struct {
	accountRepo       AccountRepository
	schedulerSnapshot *SchedulerSnapshotService
	rateLimitService  *RateLimitService
	oauthService      *CursorOAuthService
	settingService    *SettingService
	logger            *slog.Logger
}

// NewCursorGatewayService creates a new CursorGatewayService.
func NewCursorGatewayService(
	accountRepo AccountRepository,
	schedulerSnapshot *SchedulerSnapshotService,
	rateLimitService *RateLimitService,
	oauthService *CursorOAuthService,
	settingService *SettingService,
) *CursorGatewayService {
	return &CursorGatewayService{
		accountRepo:       accountRepo,
		schedulerSnapshot: schedulerSnapshot,
		rateLimitService:  rateLimitService,
		oauthService:      oauthService,
		settingService:    settingService,
		logger:            slog.Default(),
	}
}

// ──────────────────────────────────────────────────────────────────────
// Request/Response types (OpenAI-compatible)
// ──────────────────────────────────────────────────────────────────────

// cursorChatRequest is the OpenAI-compatible chat completion request.
type cursorChatRequest struct {
	Model       string              `json:"model"`
	Messages    []cursorChatMessage `json:"messages"`
	Stream      bool                `json:"stream,omitempty"`
	MaxTokens   int                 `json:"max_tokens,omitempty"`
	Temperature *float64            `json:"temperature,omitempty"`
	TopP        *float64            `json:"top_p,omitempty"`
}

// cursorChatMessage represents a single message in the chat.
type cursorChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// cursorChatResponse is a non-streaming chat completion response.
type cursorChatResponse struct {
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int64                    `json:"created"`
	Model   string                   `json:"model"`
	Choices []cursorChatChoice       `json:"choices"`
	Usage   *cursorChatResponseUsage `json:"usage,omitempty"`
}

type cursorChatChoice struct {
	Index        int                `json:"index"`
	Message      *cursorChatMessage `json:"message,omitempty"`
	Delta        *cursorChatMessage `json:"delta,omitempty"`
	FinishReason *string            `json:"finish_reason"`
}

type cursorChatResponseUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ──────────────────────────────────────────────────────────────────────
// Credential extraction
// ──────────────────────────────────────────────────────────────────────

// GetCursorCredentialsFromAccount extracts Cursor API credentials from an account (exported).
func GetCursorCredentialsFromAccount(account *Account) (CursorCredentials, error) {
	return getCursorCredentials(account)
}

// cursorMachineIDNamespace is a fixed UUID namespace for generating deterministic
// machine IDs per account. Generated once, never changes.
var cursorMachineIDNamespace = uuid.MustParse("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

// cursorConvIDNamespace is a fixed UUID namespace for generating deterministic
// conversation IDs per account+prompt, enabling Cursor prompt cache reuse.
var cursorConvIDNamespace = uuid.MustParse("b2c3d4e5-f6a7-8901-bcde-f12345678901")

// cchPattern matches the dynamic "cch=XXXXX" field in Claude Code's billing header.
var cchPattern = regexp.MustCompile(`cch=[0-9a-f]+`)

// isCursor1MModel checks if a Cursor model supports 1M context window.
// These models require max_mode=true in modelDetails to unlock 1M context,
// and use longer prefix hash (2048 chars) for better cache key discrimination.
func isCursor1MModel(cursorModel string) bool {
	// Explicit 1M suffix (e.g. claude-4-sonnet-1m)
	if strings.Contains(cursorModel, "-1m") {
		return true
	}
	// Claude 4.5/4.6 series
	if strings.HasPrefix(cursorModel, "claude-4.6-") || strings.HasPrefix(cursorModel, "claude-4.5-") {
		return true
	}
	// Gemini 2.5/3/3.1 (Flash & Pro) — all have 1M context window
	if strings.HasPrefix(cursorModel, "gemini-3.1-") ||
		strings.HasPrefix(cursorModel, "gemini-3-") ||
		strings.HasPrefix(cursorModel, "gemini-2.5-") {
		return true
	}
	// GPT-5.4 supports 1M in Max Mode
	if strings.HasPrefix(cursorModel, "gpt-5.4") {
		return true
	}
	return false
}

// stableConversationID generates a deterministic conversation ID based on
// account + prompt prefix hash, enabling Cursor prompt cache reuse.
// Same account + same system prompt → same conversationId → cache hit.
// For 1M context models, uses a longer prefix (2048 chars) to better
// distinguish between large prompts and reduce hash collisions.
func stableConversationID(accountID int64, cursorModel, promptPrefix string) string {
	// 1M context models need longer prefix for better cache key discrimination.
	prefixLen := 512
	if isCursor1MModel(cursorModel) {
		prefixLen = 2048
	}
	truncated := promptPrefix
	if len(truncated) > prefixLen {
		truncated = truncated[:prefixLen]
	}
	// Strip dynamic per-request fields that would break cache keying:
	// - "cch=XXXXX" in x-anthropic-billing-header changes every request
	truncated = cchPattern.ReplaceAllString(truncated, "cch=_")
	key := fmt.Sprintf("conv:%d:%s", accountID, truncated)
	return uuid.NewSHA1(cursorConvIDNamespace, []byte(key)).String()
}

// getCursorCredentials extracts Cursor API credentials from an account.
// When machine_id / mac_machine_id are not stored, generates deterministic
// IDs based on the account ID so each account always uses the same device
// identity (prevents "Too many computers" errors from Cursor).
func getCursorCredentials(account *Account) (CursorCredentials, error) {
	if account == nil {
		return CursorCredentials{}, errors.New("account is nil")
	}

	accessToken := account.GetCredential("access_token")
	if accessToken == "" {
		accessToken = account.GetCredential("accessToken")
	}
	if accessToken == "" {
		return CursorCredentials{}, errors.New("no access_token found in credentials")
	}

	refreshToken := account.GetCredential("refresh_token")
	if refreshToken == "" {
		refreshToken = account.GetCredential("refreshToken")
	}

	machineID := account.GetCredential("machine_id")
	if machineID == "" {
		machineID = account.GetCredential("machineId")
	}
	macMachineID := account.GetCredential("mac_machine_id")
	if macMachineID == "" {
		macMachineID = account.GetCredential("macMachineId")
	}

	// Generate deterministic machine IDs if not provided.
	// UUIDv5 = SHA1(namespace + name), so same account ID → same device ID.
	if machineID == "" {
		machineID = uuid.NewSHA1(cursorMachineIDNamespace, []byte(fmt.Sprintf("machine:%d", account.ID))).String()
	}
	if macMachineID == "" {
		macMachineID = uuid.NewSHA1(cursorMachineIDNamespace, []byte(fmt.Sprintf("mac-machine:%d", account.ID))).String()
	}

	email := account.GetCredential("email")

	return CursorCredentials{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		MachineID:    machineID,
		MacMachineID: macMachineID,
		Email:        email,
	}, nil
}

// ──────────────────────────────────────────────────────────────────────
// Model mapping
// ──────────────────────────────────────────────────────────────────────

// DefaultCursorModelMapping maps Anthropic/OpenAI model names to Cursor model identifiers.
var DefaultCursorModelMapping = map[string]string{
	// Claude 4.6 (Anthropic → Cursor)
	"claude-opus-4-6":                "claude-4.6-opus-high-thinking",
	"claude-opus-4-6-20250918":       "claude-4.6-opus-high-thinking",
	"claude-sonnet-4-6":              "claude-4.6-sonnet-medium",
	"claude-sonnet-4-6-20250514":     "claude-4.6-sonnet-medium",
	// Claude 4.5 (Anthropic → Cursor)
	"claude-opus-4-5":                "claude-4.5-opus-high-thinking",
	"claude-opus-4-5-20251101":       "claude-4.5-opus-high-thinking",
	"claude-sonnet-4-5":              "claude-4.5-sonnet",
	"claude-sonnet-4-5-20250929":     "claude-4.5-sonnet",
	"claude-haiku-4-5":               "claude-4.5-sonnet",
	"claude-haiku-4-5-20251001":      "claude-4.5-sonnet",
	// Claude 4 (Anthropic → Cursor)
	"claude-sonnet-4-20250514":       "claude-4-sonnet",
	"claude-sonnet-4":                "claude-4-sonnet",
	// Claude 3.x legacy
	"claude-3-5-sonnet-20241022":     "claude-4-sonnet",
	"claude-3.5-sonnet":              "claude-4-sonnet",
	"claude-3-5-haiku-20241022":      "claude-4-sonnet",
	"claude-3.5-haiku":               "claude-4-sonnet",
	"claude-3-haiku-20240307":        "claude-4-sonnet",
	"claude-3-opus-20240229":         "claude-4.5-opus-high",
	// Cursor passthrough (when user already uses Cursor model names)
	"claude-4.6-opus-high-thinking":  "claude-4.6-opus-high-thinking",
	"claude-4.6-opus-high":           "claude-4.6-opus-high",
	"claude-4.6-opus-max-thinking":   "claude-4.6-opus-max-thinking",
	"claude-4.6-opus-max":            "claude-4.6-opus-max",
	"claude-4.6-sonnet-medium":       "claude-4.6-sonnet-medium",
	"claude-4.5-opus-high-thinking":  "claude-4.5-opus-high-thinking",
	"claude-4.5-opus-high":           "claude-4.5-opus-high",
	"claude-4.5-sonnet":              "claude-4.5-sonnet",
	"claude-4-sonnet":                "claude-4-sonnet",
	"claude-4-sonnet-1m":             "claude-4-sonnet-1m",
	// Gemini
	"gemini-3.1-pro":                 "gemini-3.1-pro",
	"gemini-3-pro":                   "gemini-3-pro",
	"gemini-3-flash":                 "gemini-3-flash",
	// Cursor specific
	"cursor-small":                   "cursor-small",
	"default":                        "default",
}

// CursorAutoFallbackModel is the fallback model when auto-fallback is enabled
// and no mapping is found for the requested model.
const CursorAutoFallbackModel = "default"

// mapCursorModel resolves the model name to send to Cursor API.
// Falls through: account custom mapping → default mapping → auto fallback (if enabled) → passthrough.
func mapCursorModel(account *Account, requestedModel string) string {
	if account != nil {
		mapping := account.GetModelMapping()
		if len(mapping) > 0 {
			if mapped, ok := mapping[requestedModel]; ok && mapped != "" {
				return mapped
			}
			// Wildcard passthrough: "*" → "" means pass model as-is
			if _, ok := mapping["*"]; ok {
				return requestedModel
			}
		}
	}

	if mapped, ok := DefaultCursorModelMapping[requestedModel]; ok {
		return mapped
	}

	// Auto fallback: when enabled (default), unmapped models fall back to "default"
	// instead of being passed through as-is (which usually causes not_found errors).
	if account != nil && account.IsCursorAutoFallback() {
		return CursorAutoFallbackModel
	}

	return requestedModel
}

// ──────────────────────────────────────────────────────────────────────
// Forward: Main entry point
// ──────────────────────────────────────────────────────────────────────

// ForwardChatCompletions handles both streaming and non-streaming chat completions.
func (s *CursorGatewayService) ForwardChatCompletions(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
) (*ForwardResult, error) {
	prefix := fmt.Sprintf("[cursor.forward] account=%d", account.ID)

	// Parse request
	var req cursorChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("invalid request body: %w", err)
	}

	if req.Model == "" {
		return nil, errors.New("model is required")
	}

	cursorModel := mapCursorModel(account, req.Model)

	// Get credentials
	creds, err := getCursorCredentials(account)
	if err != nil {
		return nil, fmt.Errorf("%s get_credentials_failed: %w", prefix, err)
	}

	// Build conversation from messages
	conversation := buildCursorConversation(req.Messages)

	// Create SDK client
	client := NewCursorSDKClient()

	if req.Stream {
		return s.forwardStreaming(ctx, c, client, creds, account, req.Model, cursorModel, conversation, prefix)
	}
	return s.forwardNonStreaming(ctx, c, client, creds, account, req.Model, cursorModel, conversation, prefix)
}

// ──────────────────────────────────────────────────────────────────────
// Forward: Non-streaming
// ──────────────────────────────────────────────────────────────────────

// forwardNonStreaming handles a non-streaming chat completion request.
func (s *CursorGatewayService) forwardNonStreaming(
	ctx context.Context,
	c *gin.Context,
	client *CursorSDKClient,
	creds CursorCredentials,
	account *Account,
	requestedModel, cursorModel string,
	conversation string,
	prefix string,
) (*ForwardResult, error) {
	startTime := time.Now()

	eventsCh, err := client.RunChat(ctx, creds, CursorChatOptions{
		Model:          cursorModel,
		Prompt:         conversation,
		ConversationID: stableConversationID(account.ID, cursorModel, conversation),
		MaxMode:        isCursor1MModel(cursorModel),
	})
	if err != nil {
		s.logger.Error("cursor forward non-streaming failed",
			"prefix", prefix, "error", err)
		return nil, fmt.Errorf("%s forward_failed: %w", prefix, err)
	}

	// Collect all text deltas
	var fullText strings.Builder
	for event := range eventsCh {
		if event.Error != nil {
			return nil, fmt.Errorf("%s stream_error: code=%s msg=%s", prefix, event.Error.Code, event.Error.Message)
		}
		if event.TextDelta != "" {
			fullText.WriteString(event.TextDelta)
		}
	}

	responseText := fullText.String()
	completionID := "chatcmpl-" + uuid.New().String()[:8]
	finishReason := "stop"
	inputTokens := estimateTokenCount(conversation)
	outputTokens := estimateTokenCount(responseText)

	resp := cursorChatResponse{
		ID:      completionID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   requestedModel,
		Choices: []cursorChatChoice{
			{
				Index: 0,
				Message: &cursorChatMessage{
					Role:    "assistant",
					Content: responseText,
				},
				FinishReason: &finishReason,
			},
		},
		Usage: &cursorChatResponseUsage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      inputTokens + outputTokens,
		},
	}

	c.JSON(http.StatusOK, resp)

	duration := time.Since(startTime)

	return &ForwardResult{
		Model:    requestedModel,
		Stream:   false,
		Duration: duration,
		Usage: ClaudeUsage{
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
		},
	}, nil
}

// ──────────────────────────────────────────────────────────────────────
// Forward: Streaming (SSE)
// ──────────────────────────────────────────────────────────────────────

// forwardStreaming handles a streaming chat completion request, outputting SSE events.
func (s *CursorGatewayService) forwardStreaming(
	ctx context.Context,
	c *gin.Context,
	client *CursorSDKClient,
	creds CursorCredentials,
	account *Account,
	requestedModel, cursorModel string,
	conversation string,
	prefix string,
) (*ForwardResult, error) {
	startTime := time.Now()
	completionID := "chatcmpl-" + uuid.New().String()[:8]

	// Set SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	writer := c.Writer
	flusher, ok := writer.(http.Flusher)
	if !ok {
		return nil, errors.New("streaming not supported")
	}

	// Send initial role event
	initialChunk := buildSSEChunk(completionID, requestedModel, &cursorChatMessage{
		Role:    "assistant",
		Content: "",
	}, nil)
	if _, err := fmt.Fprintf(writer, "data: %s\n\n", initialChunk); err != nil {
		return nil, fmt.Errorf("failed to write initial SSE chunk: %w", err)
	}
	flusher.Flush()

	// Start Cursor streaming
	eventsCh, err := client.RunChat(ctx, creds, CursorChatOptions{
		Model:          cursorModel,
		Prompt:         conversation,
		ConversationID: stableConversationID(account.ID, cursorModel, conversation),
		MaxMode:        isCursor1MModel(cursorModel),
	})
	if err != nil {
		s.logger.Error("cursor forward streaming failed",
			"prefix", prefix, "error", err)
		// Send error in SSE format since headers are already sent
		errChunk := fmt.Sprintf(`{"error":{"message":"%s","type":"server_error"}}`, sanitizeSSEMessage(err.Error()))
		_, _ = fmt.Fprintf(writer, "data: %s\n\n", errChunk)
		_, _ = fmt.Fprintf(writer, "data: [DONE]\n\n")
		flusher.Flush()
		return nil, err
	}

	// Stream events
	var totalOutput int
	var streamError error
	for event := range eventsCh {
		if event.Error != nil {
			streamError = fmt.Errorf("code=%s msg=%s", event.Error.Code, event.Error.Message)
			errChunk := fmt.Sprintf(`{"error":{"message":"%s","type":"server_error"}}`, sanitizeSSEMessage(event.Error.Message))
			_, _ = fmt.Fprintf(writer, "data: %s\n\n", errChunk)
			flusher.Flush()
			break
		}

		if event.TextDelta != "" {
			totalOutput += len(event.TextDelta)
			chunk := buildSSEChunk(completionID, requestedModel, &cursorChatMessage{
				Content: event.TextDelta,
			}, nil)
			if _, writeErr := fmt.Fprintf(writer, "data: %s\n\n", chunk); writeErr != nil {
				streamError = writeErr
				break
			}
			flusher.Flush()
		}
	}

	// Send finish event
	finishReason := "stop"
	finishChunk := buildSSEChunk(completionID, requestedModel, nil, &finishReason)
	_, _ = fmt.Fprintf(writer, "data: %s\n\n", finishChunk)
	_, _ = fmt.Fprintf(writer, "data: [DONE]\n\n")
	flusher.Flush()

	if streamError != nil {
		s.logger.Error("cursor forward streaming error",
			"prefix", prefix, "error", streamError)
	}

	return &ForwardResult{
		Model:    requestedModel,
		Stream:   true,
		Duration: time.Since(startTime),
		Usage: ClaudeUsage{
			InputTokens:  estimateTokenCount(conversation),
			OutputTokens: estimateTokenCount(strings.Repeat("x", totalOutput)),
		},
	}, nil
}

// ──────────────────────────────────────────────────────────────────────
// Models endpoint
// ──────────────────────────────────────────────────────────────────────

// GetModels returns available Cursor models in OpenAI format.
func (s *CursorGatewayService) GetModels(ctx context.Context, account *Account) ([]byte, error) {
	creds, err := getCursorCredentials(account)
	if err != nil {
		return nil, err
	}

	client := NewCursorSDKClient()

	modelsResp, err := client.GetUsableModels(ctx, creds)
	if err != nil {
		return nil, fmt.Errorf("failed to get models: %w", err)
	}

	// Convert to OpenAI format
	type modelInfo struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	type modelsListResponse struct {
		Object string      `json:"object"`
		Data   []modelInfo `json:"data"`
	}

	resp := modelsListResponse{Object: "list"}
	for _, m := range modelsResp.Models {
		resp.Data = append(resp.Data, modelInfo{
			ID:      m.ModelID,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "cursor",
		})
	}

	return json.Marshal(resp)
}

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

// buildCursorConversation converts an array of chat messages into a single
// conversation string suitable for Cursor's UserMessage format.
func buildCursorConversation(messages []cursorChatMessage) string {
	if len(messages) == 0 {
		return ""
	}

	var buf strings.Builder
	for i, msg := range messages {
		if i > 0 {
			buf.WriteString("\n\n")
		}
		switch msg.Role {
		case "system":
			buf.WriteString("[System]\n")
			buf.WriteString(msg.Content)
		case "user":
			buf.WriteString(msg.Content)
		case "assistant":
			buf.WriteString("[Assistant]\n")
			buf.WriteString(msg.Content)
		default:
			buf.WriteString(msg.Content)
		}
	}
	return buf.String()
}

// buildSSEChunk builds a JSON string for an SSE chunk event.
func buildSSEChunk(id, model string, delta *cursorChatMessage, finishReason *string) string {
	chunk := cursorChatResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []cursorChatChoice{
			{
				Index:        0,
				Delta:        delta,
				FinishReason: finishReason,
			},
		},
	}

	data, err := json.Marshal(chunk)
	if err != nil {
		return "{}"
	}
	return string(data)
}

// estimateTokenCount provides a rough token estimate (~4 chars per token).
func estimateTokenCount(text string) int {
	return len(text) / 4
}

// sanitizeSSEMessage removes characters that would break SSE formatting.
func sanitizeSSEMessage(msg string) string {
	msg = strings.ReplaceAll(msg, "\n", " ")
	msg = strings.ReplaceAll(msg, "\r", " ")
	msg = strings.ReplaceAll(msg, "\"", "'")
	return msg
}

// WriteOpenAIChatError writes an error response in OpenAI chat format.
func (s *CursorGatewayService) WriteOpenAIChatError(c *gin.Context, statusCode int, errType, message string) {
	c.JSON(statusCode, gin.H{
		"error": gin.H{
			"message": message,
			"type":    errType,
			"code":    statusCode,
		},
	})
}

// WriteStreamingError writes an error as SSE when streaming has already started.
func (s *CursorGatewayService) WriteStreamingError(c *gin.Context, message string) {
	writer := c.Writer
	flusher, ok := writer.(http.Flusher)
	if !ok {
		return
	}

	errJSON := fmt.Sprintf(`{"error":{"message":"%s","type":"server_error"}}`, sanitizeSSEMessage(message))
	_, _ = fmt.Fprintf(writer, "data: %s\n\n", errJSON)
	_, _ = fmt.Fprintf(writer, "data: [DONE]\n\n")
	flusher.Flush()
}

// ──────────────────────────────────────────────────────────────────────
// ForwardFromMessages: Anthropic /v1/messages → Cursor
// ──────────────────────────────────────────────────────────────────────

// anthropicMessagesRequest is the subset of Anthropic /v1/messages body we need.
type anthropicMessagesRequest struct {
	Model    string          `json:"model"`
	System   json.RawMessage `json:"system,omitempty"`
	Messages []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"messages"`
	Stream   bool `json:"stream,omitempty"`
	Thinking *struct {
		Type         string `json:"type"`
		BudgetTokens int    `json:"budget_tokens,omitempty"`
	} `json:"thinking,omitempty"`
	MaxTokens int `json:"max_tokens,omitempty"`
}

// ForwardFromMessages accepts an Anthropic /v1/messages request body,
// converts it to Cursor's protocol, and writes the response back in
// Anthropic SSE or JSON format.
func (s *CursorGatewayService) ForwardFromMessages(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
) (*ForwardResult, error) {
	prefix := fmt.Sprintf("[cursor.messages] account=%d", account.ID)

	var req anthropicMessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("invalid anthropic request body: %w", err)
	}
	if req.Model == "" {
		return nil, errors.New("model is required")
	}

	cursorModel := mapCursorModel(account, req.Model)

	creds, err := getCursorCredentials(account)
	if err != nil {
		return nil, fmt.Errorf("%s get_credentials_failed: %w", prefix, err)
	}

	systemPrompt := extractAnthropicSystem(req.System)
	conversation := convertAnthropicMessagesToCursorPrompt(req.Messages)

	// Merge system prompt into conversation instead of using customSystemPrompt,
	// because some Cursor accounts/teams block "System prompt override".
	if systemPrompt != "" {
		conversation = "[System]\n" + systemPrompt + "\n\n" + conversation
		systemPrompt = "" // Don't pass as customSystemPrompt
	}

	client := NewCursorSDKClient()

	if req.Stream {
		return s.forwardStreamingAsAnthropic(ctx, c, client, creds, account, req.Model, cursorModel, systemPrompt, conversation, prefix)
	}
	return s.forwardNonStreamingAsAnthropic(ctx, c, client, creds, account, req.Model, cursorModel, systemPrompt, conversation, prefix)
}

// extractAnthropicSystem parses the Anthropic "system" field which can be
// a plain string or an array of content blocks.
func extractAnthropicSystem(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	// Try string first
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	// Try array of content blocks
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &blocks); err == nil {
		var parts []string
		for _, b := range blocks {
			if b.Type == "text" && b.Text != "" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// convertAnthropicMessagesToCursorPrompt converts Anthropic messages (with
// content blocks) into a single text prompt for Cursor.
func convertAnthropicMessagesToCursorPrompt(messages []struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}) string {
	var buf strings.Builder
	for i, msg := range messages {
		if i > 0 {
			buf.WriteString("\n\n")
		}
		text := extractAnthropicContent(msg.Content)
		switch msg.Role {
		case "user":
			buf.WriteString(text)
		case "assistant":
			buf.WriteString("[Assistant]\n")
			buf.WriteString(text)
		default:
			buf.WriteString(text)
		}
	}
	return buf.String()
}

// extractAnthropicContent parses a message's content field, which may be
// a plain string or an array of typed content blocks.
func extractAnthropicContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var blocks []struct {
		Type      string          `json:"type"`
		Text      string          `json:"text,omitempty"`
		Thinking  string          `json:"thinking,omitempty"`
		Name      string          `json:"name,omitempty"`
		Input     json.RawMessage `json:"input,omitempty"`
		ToolUseID string          `json:"tool_use_id,omitempty"`
		Content   json.RawMessage `json:"content,omitempty"`
	}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return string(raw)
	}
	var parts []string
	for _, b := range blocks {
		switch b.Type {
		case "text":
			if b.Text != "" {
				parts = append(parts, b.Text)
			}
		case "thinking":
			if b.Thinking != "" {
				parts = append(parts, b.Thinking)
			}
		case "tool_use":
			inputStr := string(b.Input)
			parts = append(parts, fmt.Sprintf("[Tool Call: %s] %s", b.Name, inputStr))
		case "tool_result":
			resultText := extractAnthropicContent(b.Content)
			parts = append(parts, fmt.Sprintf("[Tool Result: %s] %s", b.ToolUseID, resultText))
		}
	}
	return strings.Join(parts, "\n")
}

// ──────────────────────────────────────────────────────────────────────
// ForwardFromMessages: Non-streaming (Anthropic JSON response)
// ──────────────────────────────────────────────────────────────────────

func (s *CursorGatewayService) forwardNonStreamingAsAnthropic(
	ctx context.Context,
	c *gin.Context,
	client *CursorSDKClient,
	creds CursorCredentials,
	account *Account,
	requestedModel, cursorModel string,
	systemPrompt, conversation string,
	prefix string,
) (*ForwardResult, error) {
	startTime := time.Now()

	eventsCh, err := client.RunChat(ctx, creds, CursorChatOptions{
		Model:          cursorModel,
		Prompt:         conversation,
		SystemPrompt:   systemPrompt,
		ConversationID: stableConversationID(account.ID, cursorModel, conversation),
		MaxMode:        isCursor1MModel(cursorModel),
	})
	if err != nil {
		return nil, fmt.Errorf("%s forward_failed: %w", prefix, err)
	}

	var fullText strings.Builder
	var fullThinking strings.Builder
	for event := range eventsCh {
		if event.Error != nil {
			return nil, fmt.Errorf("%s stream_error: code=%s msg=%s", prefix, event.Error.Code, event.Error.Message)
		}
		if event.ThinkingDelta != "" {
			fullThinking.WriteString(event.ThinkingDelta)
		}
		if event.TextDelta != "" {
			fullText.WriteString(event.TextDelta)
		}
	}

	inputTokens := estimateTokenCount(systemPrompt + conversation)
	outputTokens := estimateTokenCount(fullText.String()) + estimateTokenCount(fullThinking.String())
	msgID := "msg_" + uuid.New().String()[:24]

	var contentBlocks []map[string]any
	if fullThinking.Len() > 0 {
		contentBlocks = append(contentBlocks, map[string]any{
			"type":     "thinking",
			"thinking": fullThinking.String(),
		})
	}
	contentBlocks = append(contentBlocks, map[string]any{
		"type": "text",
		"text": fullText.String(),
	})

	resp := map[string]any{
		"id":            msgID,
		"type":          "message",
		"role":          "assistant",
		"content":       contentBlocks,
		"model":         requestedModel,
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		},
	}
	c.JSON(http.StatusOK, resp)

	return &ForwardResult{
		Model:    requestedModel,
		Stream:   false,
		Duration: time.Since(startTime),
		Usage: ClaudeUsage{
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
		},
	}, nil
}

// ──────────────────────────────────────────────────────────────────────
// ForwardFromMessages: Streaming (Anthropic SSE)
// ──────────────────────────────────────────────────────────────────────

func (s *CursorGatewayService) forwardStreamingAsAnthropic(
	ctx context.Context,
	c *gin.Context,
	client *CursorSDKClient,
	creds CursorCredentials,
	account *Account,
	requestedModel, cursorModel string,
	systemPrompt, conversation string,
	prefix string,
) (*ForwardResult, error) {
	startTime := time.Now()
	msgID := "msg_" + uuid.New().String()[:24]
	inputTokens := estimateTokenCount(systemPrompt + conversation)

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	writer := c.Writer
	flusher, ok := writer.(http.Flusher)
	if !ok {
		return nil, errors.New("streaming not supported")
	}

	writeSSE := func(eventType string, data any) {
		jsonData, _ := json.Marshal(data)
		_, _ = fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", eventType, jsonData)
		flusher.Flush()
	}

	// message_start
	writeSSE("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            msgID,
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         requestedModel,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  inputTokens,
				"output_tokens": 0,
			},
		},
	})

	eventsCh, err := client.RunChat(ctx, creds, CursorChatOptions{
		Model:          cursorModel,
		Prompt:         conversation,
		SystemPrompt:   systemPrompt,
		ConversationID: stableConversationID(account.ID, cursorModel, conversation),
		MaxMode:        isCursor1MModel(cursorModel),
	})
	if err != nil {
		writeSSE("error", map[string]any{
			"type":  "error",
			"error": map[string]any{"type": "server_error", "message": err.Error()},
		})
		return nil, err
	}

	blockIndex := 0
	thinkingBlockOpen := false
	textBlockOpen := false
	var totalOutput int

	var streamError error
	var hasToolUse bool
	for event := range eventsCh {
		if event.Error != nil {
			streamError = fmt.Errorf("code=%s msg=%s", event.Error.Code, event.Error.Message)
			writeSSE("error", map[string]any{
				"type":  "error",
				"error": map[string]any{"type": "server_error", "message": event.Error.Message},
			})
			break
		}

		// Handle exec messages → Anthropic tool_use blocks
		if event.ExecMessage != nil {
			toolName, toolID, toolInput := cursorExecToToolUse(event.ExecMessage)
			if toolName != "" {
				// Close any open blocks first
				if thinkingBlockOpen {
					writeSSE("content_block_stop", map[string]any{
						"type":  "content_block_stop",
						"index": blockIndex,
					})
					blockIndex++
					thinkingBlockOpen = false
				}
				if textBlockOpen {
					writeSSE("content_block_stop", map[string]any{
						"type":  "content_block_stop",
						"index": blockIndex,
					})
					blockIndex++
					textBlockOpen = false
				}

				// Emit tool_use content block
				writeSSE("content_block_start", map[string]any{
					"type":  "content_block_start",
					"index": blockIndex,
					"content_block": map[string]any{
						"type":  "tool_use",
						"id":    toolID,
						"name":  toolName,
						"input": map[string]any{},
					},
				})
				// Send the full input as a single JSON delta
				inputJSON, _ := json.Marshal(toolInput)
				writeSSE("content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": blockIndex,
					"delta": map[string]any{
						"type":          "input_json_delta",
						"partial_json": string(inputJSON),
					},
				})
				writeSSE("content_block_stop", map[string]any{
					"type":  "content_block_stop",
					"index": blockIndex,
				})
				blockIndex++
				hasToolUse = true
			}
			continue
		}

		// NOTE: Cursor's thinking content lacks the cryptographic signature
		// required by Claude Code for thinking block verification.
		// We silently consume thinking deltas without forwarding them.
		if event.ThinkingDelta != "" {
			totalOutput += len(event.ThinkingDelta)
		}

		if event.ThinkingCompleted && thinkingBlockOpen {
			writeSSE("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": blockIndex,
			})
			blockIndex++
			thinkingBlockOpen = false
		}

		if event.TextDelta != "" {
			if !textBlockOpen {
				if thinkingBlockOpen {
					writeSSE("content_block_stop", map[string]any{
						"type":  "content_block_stop",
						"index": blockIndex,
					})
					blockIndex++
					thinkingBlockOpen = false
				}
				writeSSE("content_block_start", map[string]any{
					"type":          "content_block_start",
					"index":         blockIndex,
					"content_block": map[string]any{"type": "text", "text": ""},
				})
				textBlockOpen = true
			}
			totalOutput += len(event.TextDelta)
			writeSSE("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": blockIndex,
				"delta": map[string]any{"type": "text_delta", "text": event.TextDelta},
			})
		}
	}

	// Close any open blocks
	if thinkingBlockOpen {
		writeSSE("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": blockIndex,
		})
		blockIndex++
	}
	if textBlockOpen {
		writeSSE("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": blockIndex,
		})
	}

	// Set stop_reason based on whether tool_use was emitted
	stopReason := "end_turn"
	if hasToolUse {
		stopReason = "tool_use"
	}

	outputTokens := estimateTokenCount(strings.Repeat("x", totalOutput))
	writeSSE("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": outputTokens},
	})
	writeSSE("message_stop", map[string]any{"type": "message_stop"})

	if streamError != nil {
		s.logger.Error("cursor messages streaming error", "prefix", prefix, "error", streamError)
	}

	return &ForwardResult{
		Model:    requestedModel,
		Stream:   true,
		Duration: time.Since(startTime),
		Usage: ClaudeUsage{
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
		},
	}, nil
}

// cursorExecToToolUse translates a Cursor exec server message into an
// Anthropic tool_use content block. Returns (toolName, toolUseID, input) or
// empty strings if the exec message is not a tool call.
func cursorExecToToolUse(execMap map[string]any) (name string, id string, input map[string]any) {
	// shellArgs / shellStreamArgs → Bash
	if args, ok := execMap["shellArgs"].(map[string]any); ok {
		name = "Bash"
		id = extractToolCallID(args)
		input = map[string]any{"command": getStr(args, "command")}
		if desc := getStr(args, "description"); desc != "" {
			input["description"] = desc
		}
		return
	}
	if args, ok := execMap["shellStreamArgs"].(map[string]any); ok {
		name = "Bash"
		id = extractToolCallID(args)
		input = map[string]any{"command": getStr(args, "command")}
		if desc := getStr(args, "description"); desc != "" {
			input["description"] = desc
		}
		return
	}

	// writeArgs → Write
	if args, ok := execMap["writeArgs"].(map[string]any); ok {
		name = "Write"
		id = extractToolCallID(args)
		input = map[string]any{
			"file_path": getStr(args, "filePath"),
			"content":   getStr(args, "content"),
		}
		return
	}

	// readArgs → Read
	if args, ok := execMap["readArgs"].(map[string]any); ok {
		name = "Read"
		id = extractToolCallID(args)
		input = map[string]any{"file_path": getStr(args, "filePath")}
		return
	}

	// lsArgs → ListDir
	if args, ok := execMap["lsArgs"].(map[string]any); ok {
		name = "ListDir"
		id = extractToolCallID(args)
		input = map[string]any{"path": getStr(args, "path")}
		return
	}

	// grepArgs → Grep
	if args, ok := execMap["grepArgs"].(map[string]any); ok {
		name = "Grep"
		id = extractToolCallID(args)
		input = map[string]any{
			"pattern": getStr(args, "query"),
			"path":    getStr(args, "rootDir"),
		}
		if includes := getStr(args, "includes"); includes != "" {
			input["include"] = includes
		}
		return
	}

	// deleteArgs → Delete
	if args, ok := execMap["deleteArgs"].(map[string]any); ok {
		name = "Delete"
		id = extractToolCallID(args)
		input = map[string]any{"file_path": getStr(args, "filePath")}
		return
	}

	return "", "", nil
}

func extractToolCallID(args map[string]any) string {
	if id, ok := args["toolCallId"].(string); ok && id != "" {
		return id
	}
	return "toolu_" + uuid.New().String()[:24]
}

func getStr(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
