package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sovereignty-labs/nollama/internal/config"
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

func TestLoadDuplicateModelReturnsConflict(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "qwen.gguf")
	if err := os.WriteFile(modelPath, []byte("not a real gguf"), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := config.DefaultConfig()
	cfg.ModelDir = dir
	srv := NewServer(cfg, "", nil)

	// Simulate the model already being loaded by seeding a proxy route.
	srv.proxy.AddRoute("qwen.gguf", 11500)

	body := bytes.NewBufferString(`{"model":"qwen.gguf"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/load", body)
	rr := httptest.NewRecorder()
	srv.handleLoad(rr, req)

	if rr.Code != http.StatusConflict {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusConflict)
	}
	data, _ := io.ReadAll(rr.Body)
	if !strings.Contains(string(data), "already loaded") {
		t.Fatalf("response = %s", data)
	}
}

func TestLoadDuplicateModelDifferentAliasAllowed(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "qwen.gguf"), []byte("not a real gguf"), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := config.DefaultConfig()
	cfg.ModelDir = dir
	srv := NewServer(cfg, "", nil)

	// First instance already loaded under alias agent-fleet.
	srv.proxy.AddRouteWithAlias("qwen.gguf", 8001, "agent-fleet")

	// Second load with a different alias should NOT 409 on the duplicate guard.
	// (It will still fail downstream because ParseGGUF rejects the fake bytes,
	// but it must clear the HasRouteWithAlias check first.)
	body := bytes.NewBufferString(`{"model":"qwen.gguf","alias":"agent-single"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/load", body)
	rr := httptest.NewRecorder()
	srv.handleLoad(rr, req)

	if rr.Code == http.StatusConflict {
		t.Fatalf("expected dup-load guard to pass for different alias, got 409: %s", rr.Body.String())
	}
}

func TestLoadDuplicateModelSameAliasRejected(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "qwen.gguf"), []byte("not a real gguf"), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := config.DefaultConfig()
	cfg.ModelDir = dir
	srv := NewServer(cfg, "", nil)

	srv.proxy.AddRouteWithAlias("qwen.gguf", 8001, "agent-fleet")

	body := bytes.NewBufferString(`{"model":"qwen.gguf","alias":"agent-fleet"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/load", body)
	rr := httptest.NewRecorder()
	srv.handleLoad(rr, req)

	if rr.Code != http.StatusConflict {
		t.Fatalf("expected 409 for same-alias duplicate, got %d: %s", rr.Code, rr.Body.String())
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

func TestHandlePullBadMethod(t *testing.T) {
	cfg := config.DefaultConfig()
	srv := NewServer(cfg, "", nil)

	req := httptest.NewRequest(http.MethodGet, "/api/pull", nil)
	rr := httptest.NewRecorder()
	srv.handlePull(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusMethodNotAllowed)
	}
}

func TestHandlePullInvalidSpec(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelDir = t.TempDir()
	srv := NewServer(cfg, "", nil)

	req := httptest.NewRequest(http.MethodPost, "/api/pull", bytes.NewBufferString(`{"spec":"bad-spec"}`))
	rr := httptest.NewRecorder()
	srv.handlePull(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusBadRequest)
	}
}

func TestHandleRmSuccess(t *testing.T) {
	dir := t.TempDir()
	cfg := config.DefaultConfig()
	cfg.ModelDir = dir
	srv := NewServer(cfg, "", nil)

	path := filepath.Join(dir, "gpu-host.gguf")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodDelete, "/api/rm", bytes.NewBufferString(`{"model":"gpu-host"}`))
	rr := httptest.NewRecorder()
	srv.handleRm(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusOK)
	}

	var resp rmResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !resp.Deleted || resp.Filename != "gpu-host.gguf" {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("expected file to be deleted, stat err=%v", err)
	}
}

func TestHandleRmNotFound(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelDir = t.TempDir()
	srv := NewServer(cfg, "", nil)

	req := httptest.NewRequest(http.MethodDelete, "/api/rm", bytes.NewBufferString(`{"model":"missing"}`))
	rr := httptest.NewRecorder()
	srv.handleRm(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusBadRequest)
	}
}

func TestHandleRmBadMethod(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ModelDir = t.TempDir()
	srv := NewServer(cfg, "", nil)

	req := httptest.NewRequest(http.MethodGet, "/api/rm", nil)
	rr := httptest.NewRecorder()
	srv.handleRm(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusMethodNotAllowed)
	}
}

func TestHandleUploadSuccess(t *testing.T) {
	dir := t.TempDir()
	cfg := config.DefaultConfig()
	cfg.ModelDir = dir
	srv := NewServer(cfg, "", nil)

	body := []byte("abc")
	sum := sha256.Sum256(body)
	req := httptest.NewRequest(http.MethodPost, "/api/upload", bytes.NewReader(body))
	req.Header.Set("X-Filename", "gpu-host.gguf")
	req.Header.Set("X-Content-Length", "3")
	req.Header.Set("X-SHA256", hex.EncodeToString(sum[:]))
	rr := httptest.NewRecorder()

	srv.handleUpload(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusOK)
	}

	var resp uploadResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Filename != "gpu-host.gguf" || resp.Size != 3 {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if resp.SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("sha256 = %q, want %q", resp.SHA256, hex.EncodeToString(sum[:]))
	}

	data, err := os.ReadFile(filepath.Join(dir, "gpu-host.gguf"))
	if err != nil {
		t.Fatalf("read uploaded file: %v", err)
	}
	if string(data) != "abc" {
		t.Fatalf("file contents = %q, want abc", data)
	}
}

func TestHandleUploadConflict(t *testing.T) {
	dir := t.TempDir()
	cfg := config.DefaultConfig()
	cfg.ModelDir = dir
	srv := NewServer(cfg, "", nil)

	if err := os.WriteFile(filepath.Join(dir, "gpu-host.gguf"), []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/upload", bytes.NewBufferString("abc"))
	req.Header.Set("X-Filename", "gpu-host.gguf")
	req.Header.Set("X-Content-Length", "3")
	rr := httptest.NewRecorder()

	srv.handleUpload(rr, req)

	if rr.Code != http.StatusConflict {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusConflict)
	}

	var resp uploadConflictResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Error != "already exists" || resp.Filename != "gpu-host.gguf" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestHandleUploadSHA256Mismatch(t *testing.T) {
	dir := t.TempDir()
	cfg := config.DefaultConfig()
	cfg.ModelDir = dir
	srv := NewServer(cfg, "", nil)

	req := httptest.NewRequest(http.MethodPost, "/api/upload", bytes.NewBufferString("abc"))
	req.Header.Set("X-Filename", "gpu-host.gguf")
	req.Header.Set("X-Content-Length", "3")
	req.Header.Set("X-SHA256", "deadbeef")
	rr := httptest.NewRecorder()

	srv.handleUpload(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusBadRequest)
	}
	if _, err := os.Stat(filepath.Join(dir, "gpu-host.gguf.partial")); !os.IsNotExist(err) {
		t.Fatalf("expected partial file to be cleaned up, stat err=%v", err)
	}
}
