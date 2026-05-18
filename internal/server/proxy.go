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
	// RouteKey is the lookup key in the proxy's routes map. It is the lowercased
	// alias if Alias is set, otherwise the lowercased filename stem.
	RouteKey string

	// Alias is the optional per-instance route key. Empty when the route was
	// registered without an alias (key is the filename stem).
	Alias string

	// ModelName is the GGUF filename (kept verbatim for display / process
	// management). Multiple routes may share the same ModelName when they
	// differ by alias.
	ModelName string

	// BackendURL is the llama-server endpoint (e.g. http://127.0.0.1:11435).
	BackendURL *url.URL

	// Port is the llama-server port.
	Port int

	// LoadedAt records when this route was registered. Used as the grace-period
	// reference for the idle reaper.
	LoadedAt time.Time

	// LastRequest is updated on every proxied request. Initialized at load time.
	LastRequest time.Time

	// RequestCount is the total number of requests proxied through this route.
	RequestCount int64
}

// routeKeyFor returns the proxy map key for a given model filename + optional
// alias. Alias takes precedence; otherwise the filename stem is used.
func routeKeyFor(modelFilename, alias string) string {
	if alias = strings.TrimSpace(alias); alias != "" {
		return strings.ToLower(alias)
	}
	return strings.ToLower(strings.TrimSuffix(modelFilename, ".gguf"))
}

// IdleRoute describes a route that has exceeded the idle threshold.
type IdleRoute struct {
	ModelName string
	Port      int
	IdleSince time.Time
	LoadedAt  time.Time
}

// RouteStats is a snapshot of a route for metrics exposition.
type RouteStats struct {
	RouteKey     string
	Alias        string
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

// AddRoute registers a model → backend mapping keyed on the filename stem.
// For alias-keyed routes (per-instance routing), call AddRouteWithAlias.
func (p *Proxy) AddRoute(modelFilename string, port int) {
	p.AddRouteWithAlias(modelFilename, port, "")
}

// AddRouteWithAlias registers a model → backend mapping. When alias is non-empty
// the proxy keys the route on the lowercased alias; otherwise on the filename
// stem. The same model file may be registered multiple times with different
// aliases — each becomes an independent route.
func (p *Proxy) AddRouteWithAlias(modelFilename string, port int, alias string) {
	alias = strings.TrimSpace(alias)
	key := routeKeyFor(modelFilename, alias)
	backend, _ := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", port))

	now := time.Now()
	route := &Route{
		RouteKey:    key,
		Alias:       alias,
		ModelName:   modelFilename,
		BackendURL:  backend,
		Port:        port,
		LoadedAt:    now,
		LastRequest: now,
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	p.routes[key] = route
	p.rebuildAllLocked()
	p.logger.Info("route added", "model", modelFilename, "alias", alias, "port", port)
}

// RemoveRoute removes a routing entry by route key (alias if registered with
// one, otherwise filename stem). Returns true if a route was found and removed.
// Callers without a precise key should prefer RemoveRouteByPort.
func (p *Proxy) RemoveRoute(modelFilenameOrAlias string) bool {
	key := routeKeyFor(modelFilenameOrAlias, "")
	// Try direct lookup first (works when caller passes an alias or filename stem).
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.routes[key]; ok {
		delete(p.routes, key)
		p.rebuildAllLocked()
		p.logger.Info("route removed", "key", key)
		return true
	}
	// Fall back: caller passed the raw alias.
	altKey := strings.ToLower(strings.TrimSpace(modelFilenameOrAlias))
	if altKey != key {
		if _, ok := p.routes[altKey]; ok {
			delete(p.routes, altKey)
			p.rebuildAllLocked()
			p.logger.Info("route removed", "key", altKey)
			return true
		}
	}
	return false
}

// RemoveRouteByPort removes the route serving the given port. Useful for
// callers that only know the backend port (unload by port, idle reap, swap).
func (p *Proxy) RemoveRouteByPort(port int) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	for key, r := range p.routes {
		if r.Port == port {
			delete(p.routes, key)
			p.rebuildAllLocked()
			p.logger.Info("route removed", "key", key, "port", port)
			return true
		}
	}
	return false
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

// HasRoute reports whether a route exists for the given model filename.
// Equivalent to HasRouteWithAlias(modelFilename, "").
func (p *Proxy) HasRoute(modelFilename string) bool {
	return p.HasRouteWithAlias(modelFilename, "")
}

// HasRouteWithAlias reports whether a route exists keyed on the alias (when
// non-empty) or the filename stem (otherwise). Used by the duplicate-load
// guard so two loads of the same file under different aliases coexist.
func (p *Proxy) HasRouteWithAlias(modelFilename, alias string) bool {
	key := routeKeyFor(modelFilename, alias)
	p.mu.RLock()
	defer p.mu.RUnlock()
	_, ok := p.routes[key]
	return ok
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

// IdleRoutes returns routes that have been idle longer than threshold AND have
// outlived the grace period (time since LoadedAt > threshold). Returns an empty
// slice when threshold <= 0. If a route has never received a request, LoadedAt
// is used as the idle reference.
func (p *Proxy) IdleRoutes(threshold time.Duration) []IdleRoute {
	if threshold <= 0 {
		return nil
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	now := time.Now()
	var idle []IdleRoute
	for _, r := range p.all {
		if now.Sub(r.LoadedAt) <= threshold {
			continue
		}
		idleSince := r.LastRequest
		if idleSince.IsZero() {
			idleSince = r.LoadedAt
		}
		if now.Sub(idleSince) <= threshold {
			continue
		}
		idle = append(idle, IdleRoute{
			ModelName: r.ModelName,
			Port:      r.Port,
			IdleSince: idleSince,
			LoadedAt:  r.LoadedAt,
		})
	}
	return idle
}

// RouteStatsList returns a snapshot of all routes for metrics exposition.
func (p *Proxy) RouteStatsList() []RouteStats {
	p.mu.RLock()
	defer p.mu.RUnlock()

	stats := make([]RouteStats, 0, len(p.all))
	for _, r := range p.all {
		stats = append(stats, RouteStats{
			RouteKey:     r.RouteKey,
			Alias:        r.Alias,
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
