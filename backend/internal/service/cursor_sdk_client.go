package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"golang.org/x/net/http2"
)

// ============================================================
// Constants
// ============================================================

const (
	// CursorBaseURL is the primary API endpoint.
	CursorBaseURL = "https://api2.cursor.sh"

	// ConnectRPC paths
	CursorPathAgentRun       = "/agent.v1.AgentService/Run"
	CursorPathAgentRunSSE    = "/agent.v1.AgentService/RunSSE"
	CursorPathGetModels      = "/agent.v1.AgentService/GetUsableModels"
	CursorPathAuthPoll       = "/auth/poll"
	CursorPathOAuthToken     = "/oauth/token"  // correct refresh endpoint from Cursor IDE source
	CursorPathControlPing    = "/agent.v1.ControlService/Ping"
	CursorPathStripeProfie   = "/auth/full_stripe_profile"

	// DashboardService ConnectRPC paths (discovered from Cursor IDE source)
	CursorPathGetCurrentPeriodUsage = "/aiserver.v1.DashboardService/GetCurrentPeriodUsage"
	CursorPathGetPlanInfo           = "/aiserver.v1.DashboardService/GetPlanInfo"
	CursorPathGetCreditGrantsBalance = "/aiserver.v1.DashboardService/GetCreditGrantsBalance"

	// Default client version (should be kept in sync with real Cursor IDE).
	CursorDefaultClientVersion = "2.6.20"

	// Auth0 Client ID from Cursor IDE source (prod)
	CursorAuthClientID = "KbZUR41cY7W6zRSdpSUJ7I7mLYBKOCmB"

	// Heartbeat interval for BiDi streams.
	CursorHeartbeatInterval = 5 * time.Second

	// Default request timeout.
	CursorDefaultTimeout = 120 * time.Second
)

// ============================================================
// Message Types (JSON-based ConnectRPC)
// ============================================================

// --- Client → Server ---

// CursorRunRequest is the initial AgentRunRequest wrapped in an AgentClientMessage.
type CursorRunRequest struct {
	ConversationState  map[string]any      `json:"conversationState"`
	Action             *CursorAction       `json:"action"`
	ModelDetails       *CursorModelDetails `json:"modelDetails"`
	RequestedModel     *CursorModelRef     `json:"requestedModel"`
	ConversationID     string              `json:"conversationId"`
	CustomSystemPrompt string              `json:"customSystemPrompt,omitempty"`
}

type CursorAction struct {
	UserMessageAction *CursorUserMessageAction `json:"userMessageAction,omitempty"`
}

type CursorUserMessageAction struct {
	UserMessage *CursorUserMessage `json:"userMessage"`
}

type CursorUserMessage struct {
	Text string `json:"text"`
}

type CursorModelDetails struct {
	ModelID          string `json:"modelId"`
	DisplayName      string `json:"displayName,omitempty"`
	DisplayNameShort string `json:"displayNameShort,omitempty"`
}

type CursorModelRef struct {
	ModelID string `json:"modelId"`
}

// CursorClientHeartbeat is sent periodically to keep the stream alive.
type CursorClientHeartbeat struct {
	ClientHeartbeat map[string]any `json:"clientHeartbeat"`
}

// CursorExecClientMessage replies to a server exec request.
type CursorExecClientMessage struct {
	ExecClientMessage map[string]any `json:"execClientMessage"`
}

// --- Server → Client ---

// CursorServerMessage is a generic server message from the stream.
// We use map[string]any for flexibility since Cursor sends many message types.
type CursorServerMessage = map[string]any

// CursorInteractionUpdate contains streaming text, thinking, and turn info.
type CursorInteractionUpdate struct {
	TextDelta         *CursorTextDelta    `json:"textDelta,omitempty"`
	ThinkingDelta     map[string]any      `json:"thinkingDelta,omitempty"`
	ThinkingCompleted map[string]any      `json:"thinkingCompleted,omitempty"`
	TokenDelta        map[string]any      `json:"tokenDelta,omitempty"`
	TurnEnded         *CursorTurnEnded    `json:"turnEnded,omitempty"`
	StepCompleted     map[string]any      `json:"stepCompleted,omitempty"`
}

type CursorTextDelta struct {
	Text string `json:"text"`
}

type CursorTurnEnded struct {
	InputTokens      string `json:"inputTokens,omitempty"`
	OutputTokens     string `json:"outputTokens,omitempty"`
	CacheReadTokens  string `json:"cacheReadTokens,omitempty"`
	CacheWriteTokens string `json:"cacheWriteTokens,omitempty"`
}

// CursorExecServerMessage is a server-sent exec request.
type CursorExecServerMessage struct {
	ID                 int            `json:"id"`
	ExecID             string         `json:"execId"`
	RequestContextArgs map[string]any `json:"requestContextArgs,omitempty"`
	ReadArgs           map[string]any `json:"readArgs,omitempty"`
	LsArgs             map[string]any `json:"lsArgs,omitempty"`
	ShellArgs          map[string]any `json:"shellArgs,omitempty"`
	GrepArgs           map[string]any `json:"grepArgs,omitempty"`
	WriteArgs          map[string]any `json:"writeArgs,omitempty"`
	DeleteArgs         map[string]any `json:"deleteArgs,omitempty"`
	DiagnosticsArgs    map[string]any `json:"diagnosticsArgs,omitempty"`
}

// CursorUsableModel represents a model from GetUsableModels response.
type CursorUsableModel struct {
	ModelID          string `json:"modelId"`
	DisplayName      string `json:"displayName,omitempty"`
	DisplayNameShort string `json:"displayNameShort,omitempty"`
}

// CursorGetModelsResponse is the response from GetUsableModels.
type CursorGetModelsResponse struct {
	Models []CursorUsableModel `json:"models"`
}

// CursorErrorResponse represents a ConnectRPC error envelope.
type CursorErrorResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ============================================================
// Credentials
// ============================================================

// CursorCredentials holds all auth info for a Cursor account.
type CursorCredentials struct {
	AccessToken  string `json:"accessToken"`
	RefreshToken string `json:"refreshToken,omitempty"`
	MachineID    string `json:"machineId,omitempty"`
	MacMachineID string `json:"macMachineId,omitempty"`
	Email        string `json:"email,omitempty"`
}

// ============================================================
// SDK Client
// ============================================================

// CursorSDKClient communicates with the Cursor API using HTTP/2 + ConnectRPC.
type CursorSDKClient struct {
	baseURL       string
	clientVersion string
	httpClient    *http.Client
	logger        *slog.Logger
}

// NewCursorSDKClient creates a new Cursor SDK client.
func NewCursorSDKClient(logger ...*slog.Logger) *CursorSDKClient {
	// Build an HTTP/2-capable client
	transport := &http2.Transport{
		AllowHTTP: false,
	}

	l := slog.Default()
	if len(logger) > 0 && logger[0] != nil {
		l = logger[0]
	}

	return &CursorSDKClient{
		baseURL:       CursorBaseURL,
		clientVersion: CursorDefaultClientVersion,
		httpClient: &http.Client{
			Transport: transport,
			Timeout:   0, // No global timeout; per-context instead
		},
		logger: l,
	}
}

// WithBaseURL overrides the base URL (useful for testing/proxy).
func (c *CursorSDKClient) WithBaseURL(url string) *CursorSDKClient {
	c.baseURL = strings.TrimRight(url, "/")
	return c
}

// WithClientVersion overrides the client version header.
func (c *CursorSDKClient) WithClientVersion(version string) *CursorSDKClient {
	c.clientVersion = version
	return c
}

// WithHTTPClient overrides the HTTP client (e.g., with proxy).
func (c *CursorSDKClient) WithHTTPClient(client *http.Client) *CursorSDKClient {
	c.httpClient = client
	return c
}

// ============================================================
// GetUsableModels (Unary RPC)
// ============================================================

// GetUsableModels fetches the list of models available to this account.
// This is a Unary RPC, so it uses plain JSON (not envelope framing).
func (c *CursorSDKClient) GetUsableModels(ctx context.Context, creds CursorCredentials) (*CursorGetModelsResponse, error) {
	// Plain JSON body for unary RPC
	jsonBody, err := json.Marshal(map[string]any{})
	if err != nil {
		return nil, fmt.Errorf("cursor: encode GetUsableModels request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+CursorPathGetModels, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("cursor: create request: %w", err)
	}
	c.setUnaryHeaders(req, creds)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cursor: GetUsableModels request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("cursor: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("cursor: GetUsableModels HTTP %d: %s", resp.StatusCode, string(body))
	}

	// Unary response is plain JSON
	var result CursorGetModelsResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("cursor: GetUsableModels: decode: %w", err)
	}

	return &result, nil
}

// ============================================================
// Ping (Unary RPC)
// ============================================================

// Ping checks connectivity to the Cursor API.
func (c *CursorSDKClient) Ping(ctx context.Context, creds CursorCredentials) error {
	jsonBody, err := json.Marshal(map[string]any{})
	if err != nil {
		return fmt.Errorf("cursor: encode Ping: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+CursorPathControlPing, bytes.NewReader(jsonBody))
	if err != nil {
		return fmt.Errorf("cursor: create Ping request: %w", err)
	}
	c.setUnaryHeaders(req, creds)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("cursor: Ping request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("cursor: Ping HTTP %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// ============================================================
// RunChat (BiDi Streaming RPC)
// ============================================================

// CursorStreamEvent represents a parsed event from the Run stream.
type CursorStreamEvent struct {
	// Text content delta
	TextDelta string
	// Thinking content delta
	ThinkingDelta string
	// True if thinking has completed
	ThinkingCompleted bool
	// True if the turn has ended
	TurnEnded bool
	// Token usage (only set when TurnEnded is true)
	Usage *CursorTurnEnded
	// Raw message for inspection
	Raw map[string]any
	// Error from the server
	Error *CursorErrorResponse
	// ExecMessage carries a Cursor exec server message for tool_use translation.
	// When set, the gateway should translate it to an Anthropic tool_use block.
	ExecMessage map[string]any
}

// CursorChatOptions configures a chat request.
type CursorChatOptions struct {
	// Prompt is the user's message text.
	Prompt string
	// Model is the model ID (e.g. "composer-2", "claude-4.5-sonnet").
	Model string
	// ConversationID is the conversation UUID. If empty, a new one is generated.
	ConversationID string
	// SystemPrompt is an optional custom system prompt.
	SystemPrompt string
	// MaxMode enables Cursor's Max Mode, extending the context window to the
	// model's maximum (e.g. 1M tokens). Corresponds to ModelDetails.max_mode
	// in Cursor's protobuf schema (field 8).
	MaxMode bool
	// Timeout overrides the default request timeout.
	Timeout time.Duration
}

// RunChat starts a BiDi streaming chat with the Cursor API.
// It returns a channel of CursorStreamEvents. The channel is closed when
// the stream ends (turnEnded, error, or context cancellation).
func (c *CursorSDKClient) RunChat(ctx context.Context, creds CursorCredentials, opts CursorChatOptions) (<-chan CursorStreamEvent, error) {
	if opts.ConversationID == "" {
		opts.ConversationID = uuid.New().String()
	}
	if opts.Timeout == 0 {
		opts.Timeout = CursorDefaultTimeout
	}

	ctx, cancel := context.WithTimeout(ctx, opts.Timeout)

	// Build the run request message
	modelDetails := map[string]any{
		"modelId":          opts.Model,
		"displayName":      opts.Model,
		"displayNameShort": opts.Model,
	}
	if opts.MaxMode {
		modelDetails["max_mode"] = true
	}

	runMsg := map[string]any{
		"runRequest": map[string]any{
			"conversationState": map[string]any{},
			"action": map[string]any{
				"userMessageAction": map[string]any{
					"userMessage": map[string]any{
						"text": opts.Prompt,
					},
				},
			},
			"modelDetails":   modelDetails,
			"requestedModel": map[string]any{
				"modelId": opts.Model,
			},
			"conversationId": opts.ConversationID,
		},
	}

	if opts.SystemPrompt != "" {
		runMsg["runRequest"].(map[string]any)["customSystemPrompt"] = opts.SystemPrompt
	}

	// Encode as ConnectRPC frame
	frameBody, err := CursorFrameEncodeJSON(runMsg)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("cursor: encode RunChat: %w", err)
	}

	// We need pipe-based streaming: write the initial frame, then keep writing
	// heartbeats and exec replies while reading server frames.
	pr, pw := io.Pipe()

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+CursorPathAgentRun, pr)
	if err != nil {
		cancel()
		pw.Close()
		return nil, fmt.Errorf("cursor: create RunChat request: %w", err)
	}
	c.setHeaders(req, creds)
	// Streaming: disable content-length
	req.ContentLength = -1

	events := make(chan CursorStreamEvent, 64)

	// Start the HTTP request FIRST in a separate goroutine.
	// This is critical: io.Pipe is synchronous, so pw.Write blocks until
	// someone reads from pr. client.Do reads from pr, so it must be started
	// before we write to pw, otherwise we deadlock.
	respCh := make(chan *http.Response, 1)
	errCh := make(chan error, 1)
	go func() {
		resp, doErr := c.httpClient.Do(req)
		if doErr != nil {
			errCh <- doErr
			return
		}
		respCh <- resp
	}()

	go func() {
		defer cancel()
		defer close(events)
		defer pw.Close()

		// Write the initial runRequest frame (now unblocked because Do is reading)
		if _, writeErr := pw.Write(frameBody); writeErr != nil {
			c.sendEvent(events, CursorStreamEvent{Error: &CursorErrorResponse{Code: "write_error", Message: writeErr.Error()}})
			return
		}

		// Wait for response or error
		var resp *http.Response
		select {
		case resp = <-respCh:
		case doErr := <-errCh:
			c.sendEvent(events, CursorStreamEvent{Error: &CursorErrorResponse{Code: "request_error", Message: doErr.Error()}})
			return
		case <-ctx.Done():
			c.sendEvent(events, CursorStreamEvent{Error: &CursorErrorResponse{Code: "timeout", Message: ctx.Err().Error()}})
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			c.sendEvent(events, CursorStreamEvent{Error: &CursorErrorResponse{
				Code:    fmt.Sprintf("http_%d", resp.StatusCode),
				Message: string(body),
			}})
			return
		}

		// Start heartbeat writer
		var writerMu sync.Mutex
		stopHeartbeat := make(chan struct{})
		go func() {
			ticker := time.NewTicker(CursorHeartbeatInterval)
			defer ticker.Stop()
			hbFrame, _ := CursorFrameEncodeJSON(CursorClientHeartbeat{ClientHeartbeat: map[string]any{}})
			for {
				select {
				case <-ticker.C:
					writerMu.Lock()
					_, _ = pw.Write(hbFrame)
					writerMu.Unlock()
				case <-stopHeartbeat:
					return
				case <-ctx.Done():
					return
				}
			}
		}()
		defer close(stopHeartbeat)

		// Read and process frames from the server
		frameReader := NewCursorFrameReader(resp.Body)
		for {
			frame, readErr := frameReader.ReadFrame()
			if readErr != nil {
				if readErr != io.EOF {
					c.sendEvent(events, CursorStreamEvent{Error: &CursorErrorResponse{Code: "read_error", Message: readErr.Error()}})
				}
				return
			}

			// Parse the frame
			var msg map[string]any
			if err := json.Unmarshal(frame.Data, &msg); err != nil {
				c.logger.Warn("cursor: failed to parse frame", "error", err, "data", string(frame.Data))
				continue
			}

			// Check for ConnectRPC error
			if errObj, ok := msg["error"]; ok {
				if errMap, ok := errObj.(map[string]any); ok {
					c.sendEvent(events, CursorStreamEvent{Error: &CursorErrorResponse{
						Code:    fmt.Sprint(errMap["code"]),
						Message: fmt.Sprint(errMap["message"]),
					}})
					return
				}
			}

			// Handle exec server messages
			if execMsg, ok := msg["execServerMessage"]; ok {
				execMap, _ := execMsg.(map[string]any)
				if execMap != nil {
					// Platform-level messages: handle locally
					if _, ok := execMap["requestContextArgs"]; ok {
						execReply := c.handleExecMessage(execMsg)
						if execReply != nil {
							replyFrame, _ := CursorFrameEncodeJSON(execReply)
							writerMu.Lock()
							_, _ = pw.Write(replyFrame)
							writerMu.Unlock()
						}
						continue
					}
					if _, ok := execMap["diagnosticsArgs"]; ok {
						execReply := c.handleExecMessage(execMsg)
						if execReply != nil {
							replyFrame, _ := CursorFrameEncodeJSON(execReply)
							writerMu.Lock()
							_, _ = pw.Write(replyFrame)
							writerMu.Unlock()
						}
						continue
					}
					// Tool-like exec messages: forward to stream for tool_use translation,
					// but ALSO reply to Cursor so the BiDi stream doesn't hang.
					c.sendEvent(events, CursorStreamEvent{ExecMessage: execMap})
					execReply := c.handleExecMessage(execMsg)
					if execReply != nil {
						replyFrame, _ := CursorFrameEncodeJSON(execReply)
						writerMu.Lock()
						_, _ = pw.Write(replyFrame)
						writerMu.Unlock()
					}
				}
				continue
			}

			// Handle interaction updates
			if update, ok := msg["interactionUpdate"]; ok {
				if updateMap, ok := update.(map[string]any); ok {
					event := c.parseInteractionUpdate(updateMap)
					event.Raw = msg
					c.sendEvent(events, event)

					if event.TurnEnded {
						return
					}
				}
				continue
			}

			// Heartbeat and other messages — just pass raw
			if _, ok := msg["heartbeat"]; ok {
				continue // Server heartbeat, ignore
			}

			// KV messages and others — emit as raw
			c.sendEvent(events, CursorStreamEvent{Raw: msg})
		}
	}()

	return events, nil
}

// ============================================================
// RunChatSSE (Server Streaming RPC — faster for testing)
// ============================================================

// RunChatSSE uses the RunSSE (server-streaming) endpoint for a simpler, faster
// chat interaction. Unlike RunChat (BiDi), this sends the full request body
// as a single POST and reads the response stream. No heartbeats or exec
// handling is needed, making it ideal for account testing.
func (c *CursorSDKClient) RunChatSSE(ctx context.Context, creds CursorCredentials, opts CursorChatOptions) (<-chan CursorStreamEvent, error) {
	if opts.ConversationID == "" {
		opts.ConversationID = uuid.New().String()
	}
	if opts.Timeout == 0 {
		opts.Timeout = 30 * time.Second
	}

	ctx, cancel := context.WithTimeout(ctx, opts.Timeout)

	// Build the RunSSE request body (plain JSON, server-streaming ConnectRPC)
	sseModelDetails := map[string]any{
		"modelId":          opts.Model,
		"displayName":      opts.Model,
		"displayNameShort": opts.Model,
	}
	if opts.MaxMode {
		sseModelDetails["max_mode"] = true
	}

	runReq := map[string]any{
		"conversationState": map[string]any{},
		"action": map[string]any{
			"userMessageAction": map[string]any{
				"userMessage": map[string]any{
					"text": opts.Prompt,
				},
			},
		},
		"modelDetails":   sseModelDetails,
		"requestedModel": map[string]any{
			"modelId": opts.Model,
		},
		"conversationId": opts.ConversationID,
	}

	if opts.SystemPrompt != "" {
		runReq["customSystemPrompt"] = opts.SystemPrompt
	}

	// Server Streaming requires envelope-framed request (connect+json)
	frameBody, err := CursorFrameEncodeJSON(runReq)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("cursor: encode RunSSE: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+CursorPathAgentRunSSE, bytes.NewReader(frameBody))
	if err != nil {
		cancel()
		return nil, fmt.Errorf("cursor: create RunSSE request: %w", err)
	}
	// Server Streaming uses connect+json (envelope framing) for both request and response
	c.setHeaders(req, creds)

	events := make(chan CursorStreamEvent, 64)

	go func() {
		defer cancel()
		defer close(events)

		resp, err := c.httpClient.Do(req)
		if err != nil {
			c.sendEvent(events, CursorStreamEvent{Error: &CursorErrorResponse{Code: "request_error", Message: err.Error()}})
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			c.sendEvent(events, CursorStreamEvent{Error: &CursorErrorResponse{
				Code:    fmt.Sprintf("http_%d", resp.StatusCode),
				Message: string(body),
			}})
			return
		}

		// Read ConnectRPC envelope frames from the response stream
		frameReader := NewCursorFrameReader(resp.Body)
		for {
			frame, readErr := frameReader.ReadFrame()
			if readErr != nil {
				if readErr != io.EOF {
					c.sendEvent(events, CursorStreamEvent{Error: &CursorErrorResponse{Code: "read_error", Message: readErr.Error()}})
				}
				return
			}

			var msg map[string]any
			if err := json.Unmarshal(frame.Data, &msg); err != nil {
				continue
			}

			// Check for ConnectRPC error
			if errObj, ok := msg["error"]; ok {
				if errMap, ok := errObj.(map[string]any); ok {
					c.sendEvent(events, CursorStreamEvent{Error: &CursorErrorResponse{
						Code:    fmt.Sprint(errMap["code"]),
						Message: fmt.Sprint(errMap["message"]),
					}})
					return
				}
			}

			// Handle interaction updates (text deltas, turn ended, etc.)
			if update, ok := msg["interactionUpdate"]; ok {
				if updateMap, ok := update.(map[string]any); ok {
					event := c.parseInteractionUpdate(updateMap)
					event.Raw = msg
					c.sendEvent(events, event)

					if event.TurnEnded {
						return
					}
				}
				continue
			}

			// Skip heartbeats and other control messages
			if _, ok := msg["heartbeat"]; ok {
				continue
			}
		}
	}()

	return events, nil
}
// ============================================================
// Auth Helpers
// ============================================================

// CursorStripeProfile represents the response from /auth/full_stripe_profile.
type CursorStripeProfile struct {
	MembershipType string `json:"membershipType,omitempty"`
	// Raw JSON for forwarding all fields to the frontend
	Raw json.RawMessage `json:"-"`
}

// GetStripeProfile fetches the user's subscription profile.
func (c *CursorSDKClient) GetStripeProfile(ctx context.Context, creds CursorCredentials) (*CursorStripeProfile, map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+CursorPathStripeProfie, nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+creds.AccessToken)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("cursor: GetStripeProfile HTTP %d: %s", resp.StatusCode, string(body))
	}

	var profile CursorStripeProfile
	if err := json.Unmarshal(body, &profile); err != nil {
		return nil, nil, fmt.Errorf("cursor: parse profile: %w", err)
	}

	// Also return raw map for forwarding to frontend
	var raw map[string]any
	_ = json.Unmarshal(body, &raw)

	return &profile, raw, nil
}

// GetUsage fetches usage statistics from Cursor's /auth/usage endpoint.
func (c *CursorSDKClient) GetUsage(ctx context.Context, creds CursorCredentials) (map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/auth/usage", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+creds.AccessToken)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("cursor: GetUsage HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]any
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("cursor: parse usage: %w", err)
	}
	return result, nil
}

// connectRPCCall makes a generic ConnectRPC (unary) call with empty request body.
func (c *CursorSDKClient) connectRPCCall(ctx context.Context, creds CursorCredentials, path string) (map[string]any, error) {
	payload := []byte("{}")

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	c.setUnaryHeaders(req, creds)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("cursor: %s HTTP %d: %s", path, resp.StatusCode, string(body))
	}

	var result map[string]any
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("cursor: parse %s: %w", path, err)
	}
	return result, nil
}

// GetCurrentPeriodUsage fetches usage data for the current billing period.
// Response includes: billingCycleStart, billingCycleEnd, planUsage (totalSpend,
// includedSpend, bonusSpend, remaining, limit — all in cents), spendLimitUsage,
// enabled, displayMessage, autoBucketModels, etc.
func (c *CursorSDKClient) GetCurrentPeriodUsage(ctx context.Context, creds CursorCredentials) (map[string]any, error) {
	return c.connectRPCCall(ctx, creds, CursorPathGetCurrentPeriodUsage)
}

// GetPlanInfo fetches the user's plan information.
// Response includes: planInfo (planName, includedAmountCents, price, billingCycleEnd),
// nextUpgrade (tier, name, includedAmountCents, price, description).
func (c *CursorSDKClient) GetPlanInfo(ctx context.Context, creds CursorCredentials) (map[string]any, error) {
	return c.connectRPCCall(ctx, creds, CursorPathGetPlanInfo)
}

// GetCreditGrantsBalance fetches the user's credit grants balance.
// Response includes: hasCreditGrants, creditBalanceCents, totalCents, usedCents.
func (c *CursorSDKClient) GetCreditGrantsBalance(ctx context.Context, creds CursorCredentials) (map[string]any, error) {
	return c.connectRPCCall(ctx, creds, CursorPathGetCreditGrantsBalance)
}

// RefreshToken refreshes an access token using the refresh token.
// Uses the /oauth/token endpoint with standard OAuth2 refresh_token grant.
func (c *CursorSDKClient) RefreshToken(ctx context.Context, refreshToken string) (newAccessToken, newRefreshToken string, err error) {
	payload, _ := json.Marshal(map[string]string{
		"grant_type":    "refresh_token",
		"client_id":     CursorAuthClientID,
		"refresh_token": refreshToken,
	})

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+CursorPathOAuthToken, bytes.NewReader(payload))
	if err != nil {
		return "", "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("cursor: RefreshToken HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ShouldLogout bool   `json:"shouldLogout"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", "", fmt.Errorf("cursor: parse refresh response: %w", err)
	}

	if result.ShouldLogout {
		return "", "", fmt.Errorf("cursor: server requested logout")
	}

	// Cursor returns same token for both access and refresh per source code:
	// this.storeAccessRefreshToken(le.access_token, le.access_token)
	newAccess := result.AccessToken
	if newAccess == "" {
		return "", "", fmt.Errorf("cursor: empty access token in refresh response")
	}

	return newAccess, newAccess, nil
}

// ============================================================
// Internal Helpers
// ============================================================

// setHeaders applies the required Cursor API headers to a streaming request.
// Uses application/connect+json for BiDi/server-streaming RPCs (envelope framing).
func (c *CursorSDKClient) setHeaders(req *http.Request, creds CursorCredentials) {
	req.Header.Set("Content-Type", "application/connect+json")
	req.Header.Set("Connect-Protocol-Version", "1")
	req.Header.Set("Authorization", "Bearer "+creds.AccessToken)
	req.Header.Set("x-cursor-client-version", c.clientVersion)
	req.Header.Set("x-cursor-timezone", "Asia/Shanghai")
	req.Header.Set("x-request-id", uuid.New().String())

	// Always generate checksum — required for Run/RunSSE endpoints
	machineID := creds.MachineID
	macMachineID := creds.MacMachineID
	if machineID == "" {
		machineID = uuid.New().String()
	}
	if macMachineID == "" {
		macMachineID = uuid.New().String()
	}
	req.Header.Set("x-cursor-checksum", GenerateCursorChecksum(machineID, macMachineID))
}

// setUnaryHeaders applies headers for Unary ConnectRPC calls.
// Uses application/json (plain JSON body, no envelope framing).
func (c *CursorSDKClient) setUnaryHeaders(req *http.Request, creds CursorCredentials) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Connect-Protocol-Version", "1")
	req.Header.Set("Authorization", "Bearer "+creds.AccessToken)
	req.Header.Set("x-cursor-client-version", c.clientVersion)
	req.Header.Set("x-cursor-timezone", "Asia/Shanghai")
	req.Header.Set("x-request-id", uuid.New().String())
	req.Header.Set("User-Agent", "connect-es/2.0.0-rc.3")

	// Always generate checksum
	unaryMachineID := creds.MachineID
	unaryMacMachineID := creds.MacMachineID
	if unaryMachineID == "" {
		unaryMachineID = uuid.New().String()
	}
	if unaryMacMachineID == "" {
		unaryMacMachineID = uuid.New().String()
	}
	req.Header.Set("x-cursor-checksum", GenerateCursorChecksum(unaryMachineID, unaryMacMachineID))
}

// handleExecMessage handles a server exec request in headless mode.
// In headless mode (no IDE), we reject file operations and provide minimal context.
func (c *CursorSDKClient) handleExecMessage(execMsgRaw any) map[string]any {
	execMap, ok := execMsgRaw.(map[string]any)
	if !ok {
		return nil
	}

	id, _ := execMap["id"].(float64)
	execId, _ := execMap["execId"].(string)
	idInt := int(id)

	// requestContextArgs → provide minimal environment info
	if _, ok := execMap["requestContextArgs"]; ok {
		return map[string]any{
			"execClientMessage": map[string]any{
				"id":     idInt,
				"execId": execId,
				"requestContextResult": map[string]any{
					"success": map[string]any{
						"requestContext": map[string]any{
							"env": map[string]any{
								"operatingSystem": "linux",
								"defaultShell":    "bash",
							},
						},
					},
				},
			},
		}
	}

	// readArgs → fileNotFound
	if _, ok := execMap["readArgs"]; ok {
		return map[string]any{
			"execClientMessage": map[string]any{
				"id":     idInt,
				"execId": execId,
				"readResult": map[string]any{
					"fileNotFound": map[string]any{},
				},
			},
		}
	}

	// lsArgs → error
	if _, ok := execMap["lsArgs"]; ok {
		return map[string]any{
			"execClientMessage": map[string]any{
				"id":     idInt,
				"execId": execId,
				"lsResult": map[string]any{
					"error": map[string]any{
						"path":  "",
						"error": "Not available in headless mode",
					},
				},
			},
		}
	}

	// shellArgs / shellStreamArgs → rejected
	if _, ok := execMap["shellArgs"]; ok {
		return map[string]any{
			"execClientMessage": map[string]any{
				"id":     idInt,
				"execId": execId,
				"shellResult": map[string]any{
					"rejected": map[string]any{
						"reason": "Not available in headless mode",
					},
				},
			},
		}
	}
	if _, ok := execMap["shellStreamArgs"]; ok {
		return map[string]any{
			"execClientMessage": map[string]any{
				"id":     idInt,
				"execId": execId,
				"shellResult": map[string]any{
					"rejected": map[string]any{
						"reason": "Not available in headless mode",
					},
				},
			},
		}
	}

	// grepArgs → rejected
	if _, ok := execMap["grepArgs"]; ok {
		return map[string]any{
			"execClientMessage": map[string]any{
				"id":     idInt,
				"execId": execId,
				"grepResult": map[string]any{
					"error": map[string]any{
						"error": "Not available in headless mode",
					},
				},
			},
		}
	}

	// writeArgs → rejected
	if _, ok := execMap["writeArgs"]; ok {
		return map[string]any{
			"execClientMessage": map[string]any{
				"id":     idInt,
				"execId": execId,
				"writeResult": map[string]any{
					"rejected": map[string]any{
						"reason": "Not available in headless mode",
					},
				},
			},
		}
	}

	// deleteArgs → rejected
	if _, ok := execMap["deleteArgs"]; ok {
		return map[string]any{
			"execClientMessage": map[string]any{
				"id":     idInt,
				"execId": execId,
				"deleteResult": map[string]any{
					"rejected": map[string]any{
						"reason": "Not available in headless mode",
					},
				},
			},
		}
	}

	// diagnosticsArgs → empty result
	if _, ok := execMap["diagnosticsArgs"]; ok {
		return map[string]any{
			"execClientMessage": map[string]any{
				"id":     idInt,
				"execId": execId,
				"diagnosticsResult": map[string]any{},
			},
		}
	}

	// Unknown exec type — log warning and ignore
	c.logger.Warn("cursor: unknown exec server message", "keys", execMap)
	return nil
}

// parseInteractionUpdate turns an interactionUpdate map into a CursorStreamEvent.
func (c *CursorSDKClient) parseInteractionUpdate(update map[string]any) CursorStreamEvent {
	event := CursorStreamEvent{}

	// textDelta
	if td, ok := update["textDelta"]; ok {
		switch v := td.(type) {
		case string:
			event.TextDelta = v
		case map[string]any:
			if text, ok := v["text"].(string); ok {
				event.TextDelta = text
			} else if delta, ok := v["delta"].(string); ok {
				event.TextDelta = delta
			}
		}
	}

	// thinkingDelta
	if td, ok := update["thinkingDelta"]; ok {
		if tdMap, ok := td.(map[string]any); ok {
			if text, ok := tdMap["text"].(string); ok {
				event.ThinkingDelta = text
			}
		}
	}

	// thinkingCompleted
	if _, ok := update["thinkingCompleted"]; ok {
		event.ThinkingCompleted = true
	}

	// turnEnded
	if te, ok := update["turnEnded"]; ok {
		event.TurnEnded = true
		if teMap, ok := te.(map[string]any); ok {
			event.Usage = &CursorTurnEnded{
				InputTokens:      fmt.Sprint(teMap["inputTokens"]),
				OutputTokens:     fmt.Sprint(teMap["outputTokens"]),
				CacheReadTokens:  fmt.Sprint(teMap["cacheReadTokens"]),
				CacheWriteTokens: fmt.Sprint(teMap["cacheWriteTokens"]),
			}
		}
	}

	return event
}

// sendEvent sends an event to the channel without blocking.
func (c *CursorSDKClient) sendEvent(ch chan<- CursorStreamEvent, event CursorStreamEvent) {
	select {
	case ch <- event:
	default:
		c.logger.Warn("cursor: event channel full, dropping event")
	}
}
