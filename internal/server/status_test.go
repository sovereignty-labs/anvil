package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sovereignty-labs/anvil/internal/config"
)

func TestStatusGPUJSONIncludesBackend(t *testing.T) {
	payload := statusGPU{
		Index:       0,
		Name:        "AMD Radeon AI PRO R9700",
		Backend:     "rocm",
		VRAMTotalMB: 15936,
		VRAMFreeMB:  1024,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	if !strings.Contains(string(data), `"backend":"rocm"`) {
		t.Fatalf("expected backend field in %s", data)
	}
}

func TestHandleStatusIncludesAutoloadErrors(t *testing.T) {
	cfg := config.DefaultConfig()
	srv := NewServer(cfg, "", nil)
	srv.setAutoloadErrors([]autoloadError{{
		Model: "qwen.gguf",
		Alias: "turbo",
		Error: `runtime "turbo" not found in /var/lib/anvil/runtimes`,
	}})

	req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	rr := httptest.NewRecorder()
	srv.handleStatus(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusOK)
	}

	body, err := io.ReadAll(rr.Body)
	if err != nil {
		t.Fatalf("read response: %v", err)
	}
	if !strings.Contains(string(body), `"autoload_errors"`) {
		t.Fatalf("expected autoload_errors in %s", body)
	}

	var resp statusResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.AutoloadErrors) != 1 {
		t.Fatalf("autoload_errors len = %d, want 1", len(resp.AutoloadErrors))
	}
	if resp.AutoloadErrors[0].Model != "qwen.gguf" || resp.AutoloadErrors[0].Alias != "turbo" {
		t.Fatalf("unexpected autoload error payload: %+v", resp.AutoloadErrors[0])
	}
}

func TestHandleHealthIncludesAutoloadErrors(t *testing.T) {
	cfg := config.DefaultConfig()
	srv := NewServer(cfg, "", nil)
	srv.setAutoloadErrors([]autoloadError{{
		Model: "qwen.gguf",
		Error: `runtime "turbo" not found in /var/lib/anvil/runtimes`,
	}})

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rr := httptest.NewRecorder()
	srv.handleHealth(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusOK)
	}

	var resp healthResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !resp.Degraded {
		t.Fatal("expected health response to be degraded when autoload errors exist")
	}
	if len(resp.AutoloadErrors) != 1 {
		t.Fatalf("autoload_errors len = %d, want 1", len(resp.AutoloadErrors))
	}
}
