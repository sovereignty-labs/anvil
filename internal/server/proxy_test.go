package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestProxySingleModelRoutesEverything(t *testing.T) {
	// Start a fake backend
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"test","choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backend.Close()

	// Extract port from backend URL
	port := extractPort(backend.URL)

	p := NewProxy(nil)
	p.AddRoute("model-a.gguf", port)

	// Request with wrong model name — should still route to the only model
	body := `{"model":"nonexistent","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	p.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestProxyMultiModelRouting(t *testing.T) {
	// Backend A
	backendA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"source":"A"}`))
	}))
	defer backendA.Close()

	// Backend B
	backendB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"source":"B"}`))
	}))
	defer backendB.Close()

	p := NewProxy(nil)
	p.AddRoute("gemma-4-Q3.gguf", extractPort(backendA.URL))
	p.AddRoute("qwen3-Q4.gguf", extractPort(backendB.URL))

	// Request for gemma
	body := `{"model":"gemma-4-Q3","messages":[]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if !strings.Contains(rec.Body.String(), `"source":"A"`) {
		t.Errorf("expected routing to backend A, got: %s", rec.Body.String())
	}

	// Request for qwen
	body = `{"model":"qwen3-Q4","messages":[]}`
	req = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec = httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if !strings.Contains(rec.Body.String(), `"source":"B"`) {
		t.Errorf("expected routing to backend B, got: %s", rec.Body.String())
	}
}

func TestProxyFuzzyMatch(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"ok":true}`))
	}))
	defer backend.Close()

	p := NewProxy(nil)
	p.AddRoute("gemma-4-26B-A4B-Q3_K_XL.gguf", extractPort(backend.URL))
	p.AddRoute("qwen3.6-27B-IQ4_XS.gguf", 9999) // won't be hit

	// Partial match
	body := `{"model":"gemma-4-26B","messages":[]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected fuzzy match to succeed, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestProxyAliasRouting(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"routed":true}`))
	}))
	defer backend.Close()

	p := NewProxy(nil)
	p.AddRoute("qwen3.6-27B-IQ4_XS.gguf", extractPort(backend.URL))
	p.AddRoute("gemma-fake.gguf", 9999) // second model so single-model shortcut doesn't fire
	p.SetAliases(map[string]string{
		"advisor": "qwen3.6-27B-IQ4_XS",
	})

	body := `{"model":"advisor","messages":[]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected alias routing to succeed, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestProxyAliasWinsOverFuzzyMatch(t *testing.T) {
	aliasBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"source":"alias"}`))
	}))
	defer aliasBackend.Close()

	fuzzyBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"source":"fuzzy"}`))
	}))
	defer fuzzyBackend.Close()

	p := NewProxy(nil)
	p.AddRoute("qwen3.6-35B-A3B-Q4_K_S.gguf", extractPort(aliasBackend.URL))
	p.AddRoute("advisor-general.gguf", extractPort(fuzzyBackend.URL))
	p.SetAliases(map[string]string{
		"advisor": "qwen3.6-35B-A3B-Q4_K_S",
	})

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"advisor","messages":[]}`))
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected alias routing to succeed, got %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"source":"alias"`) {
		t.Fatalf("expected alias route to win over fuzzy match, got %s", rec.Body.String())
	}
}

func TestProxyNoModels(t *testing.T) {
	p := NewProxy(nil)

	body := `{"model":"anything","messages":[]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400 with no models, got %d", rec.Code)
	}
}

func TestProxyModelNotFound(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("model-a.gguf", 11111)
	p.AddRoute("model-b.gguf", 22222)

	body := `{"model":"nonexistent","messages":[]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "not found") {
		t.Errorf("expected 'not found' in error, got: %s", rec.Body.String())
	}
}

func TestProxyModelsEndpoint(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("gemma-4.gguf", 11111)
	p.AddRoute("qwen3.gguf", 22222)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp struct {
		Object string `json:"object"`
		Data   []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Object != "list" {
		t.Errorf("expected object 'list', got %s", resp.Object)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 models, got %d", len(resp.Data))
	}
}

func TestProxyHealthEndpoint(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("model.gguf", 11111)

	req := httptest.NewRequest("GET", "/health", nil)
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	body, _ := io.ReadAll(rec.Body)
	if !strings.Contains(string(body), `"models_loaded":1`) {
		t.Errorf("expected models_loaded: 1, got: %s", body)
	}
}

func TestProxyRemoveRoute(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("model-a.gguf", 11111)
	p.AddRoute("model-b.gguf", 22222)

	if p.RouteCount() != 2 {
		t.Fatalf("expected 2 routes, got %d", p.RouteCount())
	}

	p.RemoveRoute("model-a.gguf")

	if p.RouteCount() != 1 {
		t.Fatalf("expected 1 route after removal, got %d", p.RouteCount())
	}
}

func TestProxyLRURouteEmpty(t *testing.T) {
	p := NewProxy(nil)
	if lru := p.LRURoute(); lru != nil {
		t.Errorf("expected nil LRURoute with no routes, got %v", lru)
	}
}

func TestProxyLRURouteOldestWins(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("model-a.gguf", 11111)
	// AddRoute sets LastRequest to time.Now(); back-date the first one.
	p.mu.Lock()
	for _, r := range p.all {
		if r.ModelName == "model-a.gguf" {
			r.LastRequest = time.Now().Add(-1 * time.Hour)
		}
	}
	p.mu.Unlock()

	p.AddRoute("model-b.gguf", 22222)

	lru := p.LRURoute()
	if lru == nil {
		t.Fatal("expected an LRU route, got nil")
	}
	if lru.ModelName != "model-a.gguf" {
		t.Errorf("expected model-a.gguf as LRU, got %s", lru.ModelName)
	}
}

func TestProxyRouteStatsList(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("model-a.gguf", 11111)
	p.AddRoute("model-b.gguf", 22222)

	stats := p.RouteStatsList()
	if len(stats) != 2 {
		t.Fatalf("expected 2 stats entries, got %d", len(stats))
	}
	names := map[string]bool{}
	for _, st := range stats {
		names[st.ModelName] = true
		if st.LastRequest.IsZero() {
			t.Errorf("route %s has zero LastRequest", st.ModelName)
		}
		if st.RequestCount != 0 {
			t.Errorf("route %s should start with 0 requests, got %d", st.ModelName, st.RequestCount)
		}
	}
	if !names["model-a.gguf"] || !names["model-b.gguf"] {
		t.Errorf("missing expected models in stats: %v", names)
	}
}

func TestProxyRequestTrackingUpdates(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"ok":true}`))
	}))
	defer backend.Close()

	p := NewProxy(nil)
	p.AddRoute("model-a.gguf", extractPort(backend.URL))

	// Force the initial LastRequest into the past so we can detect an update.
	p.mu.Lock()
	origTime := time.Now().Add(-1 * time.Hour)
	for _, r := range p.all {
		r.LastRequest = origTime
	}
	p.mu.Unlock()

	body := `{"model":"model-a","messages":[]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	stats := p.RouteStatsList()
	if len(stats) != 1 {
		t.Fatalf("expected 1 stats entry, got %d", len(stats))
	}
	if stats[0].RequestCount != 1 {
		t.Errorf("expected RequestCount=1 after one request, got %d", stats[0].RequestCount)
	}
	if !stats[0].LastRequest.After(origTime) {
		t.Errorf("expected LastRequest to be updated past %v, got %v", origTime, stats[0].LastRequest)
	}
}

func TestAddRouteWithAlias(t *testing.T) {
	p := NewProxy(nil)
	p.AddRouteWithAlias("qwen3.6-35B-A3B.gguf", 8001, "agent-fleet")

	if !p.HasRouteWithAlias("qwen3.6-35B-A3B.gguf", "agent-fleet") {
		t.Error("expected route keyed on alias to be present")
	}
	// Filename-stem lookup should miss (the route is alias-keyed).
	if p.HasRoute("qwen3.6-35B-A3B.gguf") {
		t.Error("alias-keyed route should not be discoverable by filename stem")
	}
}

func TestDuplicateModelDifferentAlias(t *testing.T) {
	p := NewProxy(nil)
	p.AddRouteWithAlias("qwen.gguf", 8001, "agent-fleet")
	p.AddRouteWithAlias("qwen.gguf", 8002, "agent-single")

	if p.RouteCount() != 2 {
		t.Errorf("expected 2 routes, got %d", p.RouteCount())
	}
	if !p.HasRouteWithAlias("qwen.gguf", "agent-fleet") {
		t.Error("missing agent-fleet route")
	}
	if !p.HasRouteWithAlias("qwen.gguf", "agent-single") {
		t.Error("missing agent-single route")
	}
}

func TestDuplicateModelSameAliasCollides(t *testing.T) {
	p := NewProxy(nil)
	p.AddRouteWithAlias("qwen.gguf", 8001, "agent-fleet")

	if !p.HasRouteWithAlias("qwen.gguf", "agent-fleet") {
		t.Fatal("setup: first route missing")
	}
	// HasRoute with the same alias must fire — this is the dup-load guard.
	if !p.HasRouteWithAlias("qwen.gguf", "agent-fleet") {
		t.Error("HasRouteWithAlias should detect collision on same alias")
	}
}

func TestDuplicateModelNoAliasCollides(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("qwen.gguf", 8001)

	if !p.HasRoute("qwen.gguf") {
		t.Error("filename-keyed dup-load guard should fire")
	}
}

func TestRemoveRouteByAlias(t *testing.T) {
	p := NewProxy(nil)
	p.AddRouteWithAlias("qwen.gguf", 8001, "agent-fleet")
	p.AddRouteWithAlias("qwen.gguf", 8002, "agent-single")

	if !p.RemoveRoute("agent-fleet") {
		t.Error("RemoveRoute by alias key should succeed")
	}
	if p.HasRouteWithAlias("qwen.gguf", "agent-fleet") {
		t.Error("removed route should be gone")
	}
	if !p.HasRouteWithAlias("qwen.gguf", "agent-single") {
		t.Error("the other aliased instance should remain")
	}
}

func TestRemoveRouteByPort(t *testing.T) {
	p := NewProxy(nil)
	p.AddRouteWithAlias("qwen.gguf", 8001, "agent-fleet")
	p.AddRouteWithAlias("qwen.gguf", 8002, "agent-single")

	if !p.RemoveRouteByPort(8002) {
		t.Error("RemoveRouteByPort should succeed for known port")
	}
	if p.HasRouteWithAlias("qwen.gguf", "agent-single") {
		t.Error("port-8002 route should be gone")
	}
	if !p.HasRouteWithAlias("qwen.gguf", "agent-fleet") {
		t.Error("port-8001 route should remain")
	}
}

func TestProxyRoutingByAlias(t *testing.T) {
	backendFleet := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"source":"fleet"}`))
	}))
	defer backendFleet.Close()
	backendSingle := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"source":"single"}`))
	}))
	defer backendSingle.Close()

	p := NewProxy(nil)
	p.AddRouteWithAlias("qwen.gguf", extractPort(backendFleet.URL), "agent-fleet")
	p.AddRouteWithAlias("qwen.gguf", extractPort(backendSingle.URL), "agent-single")

	body := `{"model":"agent-fleet","messages":[]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)

	if !strings.Contains(rec.Body.String(), `"source":"fleet"`) {
		t.Errorf("expected fleet backend, got %s", rec.Body.String())
	}
}

func TestLoadedAtSetOnAddRoute(t *testing.T) {
	before := time.Now()
	p := NewProxy(nil)
	p.AddRoute("model-a.gguf", 11111)
	after := time.Now()

	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(p.all) != 1 {
		t.Fatalf("expected 1 route, got %d", len(p.all))
	}
	r := p.all[0]
	if r.LoadedAt.Before(before) || r.LoadedAt.After(after) {
		t.Errorf("LoadedAt %v not between %v and %v", r.LoadedAt, before, after)
	}
}

func TestIdleRoutesReturnsIdleModels(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("idle.gguf", 11111)
	p.AddRoute("active.gguf", 22222)

	// Back-date idle: loaded long ago, last request long ago.
	// Keep active fresh.
	long := -2 * time.Hour
	p.mu.Lock()
	for _, r := range p.all {
		if r.ModelName == "idle.gguf" {
			r.LoadedAt = time.Now().Add(long)
			r.LastRequest = time.Now().Add(long)
		}
	}
	p.mu.Unlock()

	idle := p.IdleRoutes(30 * time.Minute)
	if len(idle) != 1 {
		t.Fatalf("expected 1 idle route, got %d", len(idle))
	}
	if idle[0].ModelName != "idle.gguf" {
		t.Errorf("expected idle.gguf, got %s", idle[0].ModelName)
	}
}

func TestIdleRoutesGracePeriod(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("fresh.gguf", 11111)

	// LastRequest is ancient, but LoadedAt is recent (just-added) — grace.
	p.mu.Lock()
	for _, r := range p.all {
		r.LastRequest = time.Now().Add(-24 * time.Hour)
	}
	p.mu.Unlock()

	idle := p.IdleRoutes(30 * time.Minute)
	if len(idle) != 0 {
		t.Errorf("expected 0 idle routes within grace period, got %d", len(idle))
	}
}

func TestIdleRoutesNeverUsedModel(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("never-used.gguf", 11111)

	// Back-date LoadedAt past the threshold; clear LastRequest so the reaper
	// falls back to LoadedAt as the idle reference.
	p.mu.Lock()
	for _, r := range p.all {
		r.LoadedAt = time.Now().Add(-2 * time.Hour)
		r.LastRequest = time.Time{}
	}
	p.mu.Unlock()

	idle := p.IdleRoutes(30 * time.Minute)
	if len(idle) != 1 {
		t.Fatalf("expected 1 idle route (never-used past threshold), got %d", len(idle))
	}
	if idle[0].ModelName != "never-used.gguf" {
		t.Errorf("expected never-used.gguf, got %s", idle[0].ModelName)
	}
}

func TestIdleRoutesNoIdleModels(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("a.gguf", 11111)
	p.AddRoute("b.gguf", 22222)

	idle := p.IdleRoutes(30 * time.Minute)
	if len(idle) != 0 {
		t.Errorf("expected 0 idle routes, got %d", len(idle))
	}
}

func TestIdleRoutesZeroThreshold(t *testing.T) {
	p := NewProxy(nil)
	p.AddRoute("ancient.gguf", 11111)
	p.mu.Lock()
	for _, r := range p.all {
		r.LoadedAt = time.Now().Add(-24 * time.Hour)
		r.LastRequest = time.Now().Add(-24 * time.Hour)
	}
	p.mu.Unlock()

	if got := p.IdleRoutes(0); len(got) != 0 {
		t.Errorf("expected zero-threshold to disable selection, got %d", len(got))
	}
}

// extractPort pulls the port number from an httptest.Server URL.
func extractPort(rawURL string) int {
	parts := strings.Split(rawURL, ":")
	port := 0
	for i := len(parts[len(parts)-1]) - 1; i >= 0; i-- {
		c := parts[len(parts)-1][i]
		if c < '0' || c > '9' {
			break
		}
	}
	// Simple parse
	for _, c := range parts[len(parts)-1] {
		if c >= '0' && c <= '9' {
			port = port*10 + int(c-'0')
		}
	}
	return port
}
