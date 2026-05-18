package federation

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"time"
)

// Client talks to a remote nollama management API.
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NodeStatus mirrors internal/server.statusResponse.
type NodeStatus struct {
	Models []StatusModel `json:"models"`
	Node   StatusNode    `json:"node"`
}

// StatusResponse is the public name used by callers.
type StatusResponse = NodeStatus

// StatusModel mirrors internal/server.statusModel.
type StatusModel struct {
	Name          string `json:"name"`
	Alias         string `json:"alias,omitempty"`
	Port          int    `json:"port"`
	GPU           string `json:"gpu"`
	PID           int    `json:"pid"`
	UptimeSeconds int64  `json:"uptime_seconds"`
}

// StatusNode mirrors internal/server.statusNode.
type StatusNode struct {
	GPUs       []StatusGPU `json:"gpus"`
	RAMTotalMB uint64      `json:"ram_total_mb"`
	RAMFreeMB  uint64      `json:"ram_free_mb"`
}

// StatusGPU mirrors internal/server.statusGPU.
type StatusGPU struct {
	Index       int    `json:"index"`
	Name        string `json:"name"`
	VRAMTotalMB uint64 `json:"vram_total_mb"`
	VRAMFreeMB  uint64 `json:"vram_free_mb"`
}

// LoadRequest mirrors internal/server.loadRequest.
type LoadRequest struct {
	Model string         `json:"model"`
	GPU   *int           `json:"gpu,omitempty"`
	CPU   bool           `json:"cpu,omitempty"`
	Flags map[string]any `json:"flags,omitempty"`
	Swap  bool           `json:"swap,omitempty"`
	Port  int            `json:"port,omitempty"`
	Alias string         `json:"alias,omitempty"`
}

// LoadResult mirrors internal/server.loadResponse.
type LoadResult struct {
	Status string `json:"status"`
	Model  string `json:"model"`
	Port   int    `json:"port"`
	Device string `json:"device"`
	PID    int    `json:"pid"`
}

// LoadResponse is the public name used by callers.
type LoadResponse = LoadResult

// PullResponse mirrors the server upload response for remote pulls.
type PullResponse struct {
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
}

type RmResponse struct {
	Filename string `json:"filename"`
	Deleted  bool   `json:"deleted"`
}

// UnloadRequest mirrors internal/server.unloadRequest.
type UnloadRequest struct {
	Model string `json:"model"`
}

// UnloadResponse mirrors internal/server.unloadResponse.
type UnloadResponse struct {
	Status string `json:"status"`
}

// ModelInfo mirrors internal/server.modelSummary.
type ModelInfo struct {
	Name          string `json:"name"`
	SizeBytes     int64  `json:"size_bytes"`
	SizeHuman     string `json:"size_human"`
	Arch          string `json:"arch"`
	Quant         string `json:"quant"`
	ContextLength uint64 `json:"context_length"`
}

// ModelsResponse mirrors internal/server.modelsResponse.
type ModelsResponse struct {
	Models []ModelInfo `json:"models"`
}

type apiErrorResponse struct {
	Error string `json:"error"`
}

// NewClient creates a client with a 30-second timeout.
func NewClient(baseURL string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(strings.TrimSpace(baseURL), "/"),
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Status fetches /api/status.
func (c *Client) Status() (*StatusResponse, error) {
	var resp StatusResponse
	if err := c.doJSON(http.MethodGet, "/api/status", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Load posts to /api/load.
func (c *Client) Load(req LoadRequest) (*LoadResponse, error) {
	var resp LoadResponse
	if err := c.doJSON(http.MethodPost, "/api/load", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Unload posts to /api/unload.
func (c *Client) Unload(model string) error {
	return c.doJSON(http.MethodPost, "/api/unload", UnloadRequest{Model: model}, nil)
}

// Models fetches /api/models.
func (c *Client) Models() (*ModelsResponse, error) {
	var resp ModelsResponse
	if err := c.doJSON(http.MethodGet, "/api/models", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Pull triggers a HuggingFace pull on the remote node.
func (c *Client) Pull(spec string) (*PullResponse, error) {
	endpoint, err := c.endpoint("/api/pull")
	if err != nil {
		return nil, err
	}

	reqBody := struct {
		Spec string `json:"spec"`
	}{Spec: spec}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("encoding request for /api/pull: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("building request for /api/pull: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Pulls can take minutes for large models, so use a dedicated client with no timeout.
	client := &http.Client{Timeout: 0}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response from /api/pull: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, requestError("/api/pull", resp.StatusCode, data)
	}

	var out PullResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("decoding response from /api/pull: %w", err)
	}
	return &out, nil
}

// Rm deletes a model from the remote node.
func (c *Client) Rm(model string) error {
	return c.doJSON(http.MethodPost, "/api/rm", struct {
		Model string `json:"model"`
	}{Model: model}, nil)
}

// CheckModelExists checks whether the remote node already has a model file.
func (c *Client) CheckModelExists(filename string) (bool, int64, error) {
	models, err := c.Models()
	if err != nil {
		return false, 0, err
	}

	name := filepath.Base(filename)
	for _, model := range models.Models {
		if model.Name == name {
			return true, model.SizeBytes, nil
		}
	}

	return false, 0, nil
}

// UploadModel streams a GGUF file to /api/upload.
func (c *Client) UploadModel(filename string, reader io.Reader, size int64, sha256hex string) (string, error) {
	endpoint, err := c.endpoint("/api/upload")
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest(http.MethodPost, endpoint, reader)
	if err != nil {
		return "", fmt.Errorf("building upload request for %s: %w", filename, err)
	}
	req.Header.Set("X-Filename", filepath.Base(filename))
	req.Header.Set("X-Content-Length", fmt.Sprintf("%d", size))
	if sha256hex != "" {
		req.Header.Set("X-SHA256", sha256hex)
	}
	if size >= 0 {
		req.ContentLength = size
	}

	// Uploads may take minutes for multi-GB GGUFs; always use a no-timeout client.
	client := &http.Client{Timeout: 0}

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return "", requestError("/api/upload", resp.StatusCode, data)
	}

	var payload struct {
		SHA256 string `json:"sha256"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return "", fmt.Errorf("decoding response from /api/upload: %w", err)
	}

	return payload.SHA256, nil
}

// Health fetches /health and requires HTTP 200.
func (c *Client) Health() error {
	resp, err := c.doRequest(http.MethodGet, "/health", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned %s", resp.Status)
	}

	return nil
}

func (c *Client) doJSON(method, path string, body any, out any) error {
	resp, err := c.doRequest(method, path, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("reading response from %s: %w", path, err)
	}

	if resp.StatusCode != http.StatusOK {
		return requestError(path, resp.StatusCode, data)
	}
	if out == nil || len(data) == 0 {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("decoding response from %s: %w", path, err)
	}
	return nil
}

func (c *Client) doRequest(method, path string, body any) (*http.Response, error) {
	endpoint, err := c.endpoint(path)
	if err != nil {
		return nil, err
	}

	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("encoding request for %s: %w", path, err)
		}
		reader = bytes.NewReader(data)
	}

	req, err := http.NewRequest(method, endpoint, reader)
	if err != nil {
		return nil, fmt.Errorf("building request for %s: %w", path, err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	client := c.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func (c *Client) endpoint(path string) (string, error) {
	base := strings.TrimRight(strings.TrimSpace(c.BaseURL), "/")
	if base == "" {
		return "", fmt.Errorf("base URL is required")
	}
	return base + "/" + strings.TrimLeft(path, "/"), nil
}

func requestError(path string, statusCode int, body []byte) error {
	var apiErr apiErrorResponse
	if len(body) > 0 && json.Unmarshal(body, &apiErr) == nil && apiErr.Error != "" {
		return fmt.Errorf("%s: %s", http.StatusText(statusCode), apiErr.Error)
	}
	if len(body) > 0 {
		return fmt.Errorf("%s: %s", http.StatusText(statusCode), strings.TrimSpace(string(body)))
	}
	return fmt.Errorf("%s from %s", http.StatusText(statusCode), path)
}
