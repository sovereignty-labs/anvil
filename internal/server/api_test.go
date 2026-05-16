package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hirdforge/nollama/internal/config"
)

func TestHandleModels(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "Qwen3.6-27B-IQ4_XS.gguf")
	if err := os.WriteFile(modelPath, []byte("not a real gguf"), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := config.DefaultConfig()
	cfg.ModelDir = dir
	srv := NewServer(cfg, "", nil)

	req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
	rr := httptest.NewRecorder()
	srv.handleModels(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusOK)
	}

	var resp modelsResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Models) != 1 {
		t.Fatalf("models len = %d, want 1", len(resp.Models))
	}
	if resp.Models[0].Name != "Qwen3.6-27B-IQ4_XS.gguf" {
		t.Fatalf("model name = %q", resp.Models[0].Name)
	}
}

func TestHandleLoadMissingModel(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelDir = t.TempDir()
	srv := NewServer(cfg, "", nil)

	body := bytes.NewBufferString(`{"model":"missing.gguf"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/load", body)
	rr := httptest.NewRecorder()
	srv.handleLoad(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusBadRequest)
	}
	data, _ := io.ReadAll(rr.Body)
	if !strings.Contains(string(data), "model not found") {
		t.Fatalf("response = %s", data)
	}
}

func TestHandleUnloadNotRunning(t *testing.T) {
	cfg := config.DefaultConfig()
	srv := NewServer(cfg, "", nil)

	body := bytes.NewBufferString(`{"model":"missing.gguf"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/unload", body)
	rr := httptest.NewRecorder()
	srv.handleUnload(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusBadRequest)
	}
	data, _ := io.ReadAll(rr.Body)
	if !strings.Contains(string(data), "no running process found") {
		t.Fatalf("response = %s", data)
	}
}

func TestHandleLoadBadMethod(t *testing.T) {
	cfg := config.DefaultConfig()
	srv := NewServer(cfg, "", nil)

	req := httptest.NewRequest(http.MethodGet, "/api/load", nil)
	rr := httptest.NewRecorder()
	srv.handleLoad(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusMethodNotAllowed)
	}
}

func TestHandleUnloadBadMethod(t *testing.T) {
	cfg := config.DefaultConfig()
	srv := NewServer(cfg, "", nil)

	req := httptest.NewRequest(http.MethodGet, "/api/unload", nil)
	rr := httptest.NewRecorder()
	srv.handleUnload(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusMethodNotAllowed)
	}
}
