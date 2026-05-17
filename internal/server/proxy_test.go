package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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
