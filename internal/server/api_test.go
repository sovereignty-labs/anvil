package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sovereignty-labs/nollama/internal/config"
	"github.com/sovereignty-labs/nollama/internal/model"
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

	body := bytes.NewBufferString(`{"model":"qwen.gguf","port":45555}`)
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

func TestHandleLoadDoesNotRegisterRouteUntilReady(t *testing.T) {
	dir := t.TempDir()
	writeTestGGUF(t, dir, "qwen.gguf")

	mockScript := filepath.Join(dir, "crash-llama-server")
	if err := os.WriteFile(mockScript, []byte("#!/bin/sh\necho 'unknown model architecture: qwen3.6' 1>&2\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	cfg := config.DefaultConfig()
	cfg.ModelDir = dir
	cfg.LlamaServer = mockScript
	srv := NewServer(cfg, "", nil)
	srv.procMgr.SetLogDir(dir)

	body := bytes.NewBufferString(`{"model":"qwen.gguf","port":45555}`)
	req := httptest.NewRequest(http.MethodPost, "/api/load", body)
	rr := httptest.NewRecorder()
	srv.handleLoad(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusInternalServerError)
	}
	data, _ := io.ReadAll(rr.Body)
	if !strings.Contains(string(data), "crashed during model loading") {
		t.Fatalf("response = %s", data)
	}
	if srv.proxy.RouteCount() != 0 {
		t.Fatalf("expected no route to be registered on load failure, got %d", srv.proxy.RouteCount())
	}
}

func TestLoadModelUsesEntryRuntimeBackendAndBinary(t *testing.T) {
	root := t.TempDir()
	xdgConfigHome := filepath.Join(root, "config")
	runtimesDir := filepath.Join(root, "runtimes")
	modelDir := filepath.Join(root, "models")

	if err := os.MkdirAll(filepath.Join(xdgConfigHome, "nollama"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(xdgConfigHome, "nollama", "config.yaml"), []byte("runtimes_dir: "+runtimesDir+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("XDG_CONFIG_HOME", xdgConfigHome)

	rocmDir := filepath.Join(runtimesDir, "llama-rocm")
	cudaDir := filepath.Join(runtimesDir, "llama-b9375")
	for _, dir := range []string{rocmDir, cudaDir, modelDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	buildTestRuntimeBinary(t, rocmDir, "rocm")
	buildTestRuntimeBinary(t, cudaDir, "cuda")
	if err := os.WriteFile(filepath.Join(rocmDir, "backend"), []byte("rocm\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cudaDir, "backend"), []byte("cuda\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeTestGGUF(t, modelDir, "qwen.gguf")

	cfg := config.DefaultConfig()
	cfg.ModelDir = modelDir
	cfg.LlamaServer = filepath.Join(cudaDir, "llama-server")
	srv := NewServer(cfg, "", nil)
	srv.procMgr.SetLogDir(root)

	gpu := 0
	entry := config.AutoloadEntry{
		Model:   "qwen.gguf",
		Runtime: "llama-rocm",
		GPU:     &gpu,
	}

	port, err := srv.loadModel(entry, nil)
	if err != nil {
		t.Fatalf("loadModel() error: %v", err)
	}
	t.Cleanup(func() {
		_, _ = srv.procMgr.StopByPort(port)
	})

	proc := srv.procMgr.GetByPort(port)
	if proc == nil {
		t.Fatal("expected running process to be tracked")
	}
	if proc.GPUIndex != "rocm:0" {
		t.Fatalf("GPUIndex = %q, want rocm:0", proc.GPUIndex)
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

func TestBuildAutoloadEnvMapsVkDevice(t *testing.T) {
	entry := config.AutoloadEntry{
		Env: map[string]string{
			"FOO": "bar",
		},
	}
	merged := map[string]interface{}{
		"ctx-size":  65536,
		"vk-device": "1",
	}

	env, flags := buildAutoloadEnv(entry, merged)
	if env["FOO"] != "bar" {
		t.Fatalf("env FOO = %q, want bar", env["FOO"])
	}
	if env["GGML_VK_DEVICE"] != "1" {
		t.Fatalf("env GGML_VK_DEVICE = %q, want 1", env["GGML_VK_DEVICE"])
	}
	if _, ok := flags["vk-device"]; ok {
		t.Fatal("vk-device flag should be removed before llama-server argv is built")
	}
}

func writeTestGGUF(t *testing.T, dir, name string) string {
	t.Helper()

	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	order := binary.LittleEndian
	if _, err := f.Write([]byte("GGUF")); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, order, uint32(3)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, order, uint64(0)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, order, uint64(3)); err != nil {
		t.Fatal(err)
	}

	kvs := []struct {
		key   string
		typ   uint32
		value any
	}{
		{key: "general.architecture", typ: model.GGUFTypeString, value: "qwen2"},
		{key: "general.context_length", typ: model.GGUFTypeUint64, value: uint64(4096)},
		{key: "general.file_type", typ: model.GGUFTypeUint32, value: uint32(15)},
	}

	for _, kv := range kvs {
		if err := binary.Write(f, order, uint64(len(kv.key))); err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write([]byte(kv.key)); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(f, order, kv.typ); err != nil {
			t.Fatal(err)
		}
		switch v := kv.value.(type) {
		case string:
			if err := binary.Write(f, order, uint64(len(v))); err != nil {
				t.Fatal(err)
			}
			if _, err := f.Write([]byte(v)); err != nil {
				t.Fatal(err)
			}
		case uint64:
			if err := binary.Write(f, order, v); err != nil {
				t.Fatal(err)
			}
		case uint32:
			if err := binary.Write(f, order, v); err != nil {
				t.Fatal(err)
			}
		}
	}

	return path
}

func buildTestRuntimeBinary(t *testing.T, dir, label string) string {
	t.Helper()

	source := filepath.Join(dir, "mock_runtime.go")
	binary := filepath.Join(dir, "llama-server")
	code := fmt.Sprintf(`package main

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
)

func main() {
	label := %q
	fmt.Fprintf(os.Stderr, "runtime=%%s\n", label)
	port := 11434
	for i := 1; i+1 < len(os.Args); i++ {
		if os.Args[i] == "--port" {
			if parsed, err := strconv.Atoi(os.Args[i+1]); err == nil {
				port = parsed
			}
		}
	}
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprintln(w, "ok")
	})
	_ = http.ListenAndServe("127.0.0.1:"+strconv.Itoa(port), nil)
}
`, label)
	if err := os.WriteFile(source, []byte(code), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "build", "-o", binary, source)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build test runtime %s: %v\n%s", label, err, out)
	}
	if err := os.Remove(source); err != nil {
		t.Fatal(err)
	}
	return binary
}
