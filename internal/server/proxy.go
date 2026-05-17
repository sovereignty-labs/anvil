// Package server implements the nollama serve daemon and API proxy.
package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/hirdforge/nollama/internal/model"
)

// Route maps a model name to a llama-server backend.
type Route struct {
	// ModelName is the GGUF filename (without .gguf extension).
	ModelName string

	// BackendURL is the llama-server endpoint (e.g. http://127.0.0.1:11435).
	BackendURL *url.URL

	// Port is the llama-server port.
	Port int

	// LastRequest is updated on every proxied request. Initialized at load time.
	LastRequest time.Time

	// RequestCount is the total number of requests proxied through this route.
	RequestCount int64
}

// RouteStats is a snapshot of a route for metrics exposition.
type RouteStats struct {
	ModelName    string
	Port         int
	LastRequest  time.Time
	RequestCount int64
}

// Proxy is an HTTP reverse proxy that routes requests by model name.
type Proxy struct {
	mu     sync.RWMutex
	routes map[string]*Route // model stem (lowercase) → route
	all    []*Route          // ordered list of all routes

	// Aliases maps friendly names to model filenames.
	aliases map[string]string

	logger *slog.Logger
}

// NewProxy creates a new API proxy.
func NewProxy(logger *slog.Logger) *Proxy {
	if logger == nil {
		logger = slog.Default()
	}
	return &Proxy{
		routes:  make(map[string]*Route),
		aliases: make(map[string]string),
		logger:  logger,
	}
}

// AddRoute registers a model → backend mapping.
func (p *Proxy) AddRoute(modelFilename string, port int) {
	stem := strings.ToLower(strings.TrimSuffix(modelFilename, ".gguf"))
	backend, _ := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", port))

	route := &Route{
		ModelName:   modelFilename,
		BackendURL:  backend,
		Port:        port,
		LastRequest: time.Now(),
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	p.routes[stem] = route
	p.rebuildAllLocked()
	p.logger.Info("route added", "model", modelFilename, "port", port)
}

// RemoveRoute removes a model's routing entry.
func (p *Proxy) RemoveRoute(modelFilename string) {
	stem := strings.ToLower(strings.TrimSuffix(modelFilename, ".gguf"))

	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.routes, stem)
	p.rebuildAllLocked()
	p.logger.Info("route removed", "model", modelFilename)
}

// SetAliases updates the alias map.
func (p *Proxy) SetAliases(aliases map[string]string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.aliases = make(map[string]string, len(aliases))
	for k, v := range aliases {
		p.aliases[strings.ToLower(k)] = v
	}
}

// RouteCount returns the number of active routes.
func (p *Proxy) RouteCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.routes)
}

// LRURoute returns the route with the oldest LastRequest, or nil if no routes exist.
func (p *Proxy) LRURoute() *Route {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.all) == 0 {
		return nil
	}

	var lru *Route
	for _, r := range p.all {
		if lru == nil || r.LastRequest.Before(lru.LastRequest) {
			lru = r
		}
	}
	return lru
}

// RouteStatsList returns a snapshot of all routes for metrics exposition.
func (p *Proxy) RouteStatsList() []RouteStats {
	p.mu.RLock()
	defer p.mu.RUnlock()

	stats := make([]RouteStats, 0, len(p.all))
	for _, r := range p.all {
		stats = append(stats, RouteStats{
			ModelName:    r.ModelName,
			Port:         r.Port,
			LastRequest:  r.LastRequest,
			RequestCount: r.RequestCount,
		})
	}
	return stats
}

func (p *Proxy) rebuildAllLocked() {
	p.all = make([]*Route, 0, len(p.routes))
	for _, r := range p.routes {
		p.all = append(p.all, r)
	}
}

// resolve finds the backend for a model name.
// When only one model is loaded, returns it regardless of name.
func (p *Proxy) resolve(modelName string) (*Route, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	// Single model loaded — route everything to it
	if len(p.all) == 1 {
		return p.all[0], nil
	}

	if len(p.all) == 0 {
		return nil, fmt.Errorf("no models loaded")
	}

	name := strings.ToLower(strings.TrimSuffix(modelName, ".gguf"))

	// Check aliases first
	if target, ok := p.aliases[name]; ok {
		name = strings.ToLower(strings.TrimSuffix(target, ".gguf"))
	}

	// Exact stem match
	if r, ok := p.routes[name]; ok {
		return r, nil
	}

	// Fuzzy match
	available := make([]string, 0, len(p.routes))
	for stem := range p.routes {
		available = append(available, stem)
	}
	if match := model.FuzzyMatchModel(name, available); match != "" {
		if r, ok := p.routes[match]; ok {
			return r, nil
		}
	}

	// List available models in error
	names := make([]string, 0, len(p.routes))
	for _, r := range p.all {
		names = append(names, r.ModelName)
	}
	return nil, fmt.Errorf("model %q not found. loaded: %s", modelName, strings.Join(names, ", "))
}

// ServeHTTP handles all incoming API requests.
func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.URL.Path == "/v1/models" || r.URL.Path == "/v1/models/":
		p.handleModels(w, r)
	case r.URL.Path == "/health":
		p.handleHealth(w, r)
	case strings.HasPrefix(r.URL.Path, "/v1/"):
		p.handleProxy(w, r)
	default:
		// Try to proxy anything else too — llama-server may handle it
		p.handleProxy(w, r)
	}
}

// handleProxy reads the model field from the request body and proxies to the right backend.
func (p *Proxy) handleProxy(w http.ResponseWriter, r *http.Request) {
	// Read body to extract model name
	var bodyBytes []byte
	var err error
	if r.Body != nil {
		bodyBytes, err = io.ReadAll(r.Body)
		r.Body.Close()
		if err != nil {
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return
		}
	}

	// Extract model name from JSON body
	modelName := ""
	if len(bodyBytes) > 0 {
		var req struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(bodyBytes, &req); err == nil {
			modelName = req.Model
		}
	}

	// Resolve route
	route, err := p.resolve(modelName)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Update request tracking
	p.mu.Lock()
	route.LastRequest = time.Now()
	route.RequestCount++
	p.mu.Unlock()

	// Restore body for proxying
	r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	r.ContentLength = int64(len(bodyBytes))

	// Create reverse proxy
	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = route.BackendURL.Scheme
			req.URL.Host = route.BackendURL.Host
			req.Host = route.BackendURL.Host
		},
		// Flush immediately for streaming (SSE) responses
		FlushInterval: 100 * time.Millisecond,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			p.logger.Error("proxy error",
				"model", route.ModelName,
				"port", route.Port,
				"error", err,
			)
			http.Error(w, fmt.Sprintf("backend error (model %s): %v", route.ModelName, err), http.StatusBadGateway)
		},
	}

	p.logger.Debug("proxying request",
		"path", r.URL.Path,
		"model", route.ModelName,
		"port", route.Port,
	)

	proxy.ServeHTTP(w, r)
}

// handleModels returns an OpenAI-compatible model list aggregated from all loaded models.
func (p *Proxy) handleModels(w http.ResponseWriter, r *http.Request) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	type modelObj struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}

	models := make([]modelObj, 0, len(p.all))
	for _, route := range p.all {
		stem := strings.TrimSuffix(route.ModelName, ".gguf")
		models = append(models, modelObj{
			ID:      stem,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "nollama",
		})
	}

	resp := struct {
		Object string     `json:"object"`
		Data   []modelObj `json:"data"`
	}{
		Object: "list",
		Data:   models,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleHealth returns nollama health status.
func (p *Proxy) handleHealth(w http.ResponseWriter, r *http.Request) {
	p.mu.RLock()
	count := len(p.all)
	p.mu.RUnlock()

	resp := struct {
		Status string `json:"status"`
		Models int    `json:"models_loaded"`
	}{
		Status: "ok",
		Models: count,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
