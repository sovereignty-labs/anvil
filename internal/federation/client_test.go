package federation

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestClientStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/api/status" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(StatusResponse{
			Models: []StatusModel{{
				Name:          "gemma-4-q3.gguf",
				Port:          11435,
				GPU:           "cuda:0",
				PID:           1234,
				UptimeSeconds: 3600,
			}},
			Node: StatusNode{
				GPUs: []StatusGPU{{
					Index:       0,
					Name:        "RTX 3090 24GB",
					VRAMTotalMB: 24576,
					VRAMFreeMB:  16384,
				}},
				RAMTotalMB: 32768,
				RAMFreeMB:  20480,
			},
		})
	}))
	defer srv.Close()

	client := NewClient(srv.URL)
	resp, err := client.Status()
	if err != nil {
		t.Fatalf("Status failed: %v", err)
	}
	if len(resp.Models) != 1 {
		t.Fatalf("models len = %d, want 1", len(resp.Models))
	}
	if resp.Models[0].Name != "gemma-4-q3.gguf" {
		t.Fatalf("model name = %q", resp.Models[0].Name)
	}
	if len(resp.Node.GPUs) != 1 || resp.Node.GPUs[0].Name != "RTX 3090 24GB" {
		t.Fatalf("unexpected node GPU payload: %+v", resp.Node.GPUs)
	}
}

func TestClientModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/api/models" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(ModelsResponse{
			Models: []ModelInfo{{
				Name:          "gemma-4-q3.gguf",
				SizeBytes:     1024,
				SizeHuman:     "1 KB",
				Arch:          "gemma4",
				Quant:         "Q3_K_XL",
				ContextLength: 131072,
			}},
		})
	}))
	defer srv.Close()

	client := NewClient(srv.URL)
	resp, err := client.Models()
	if err != nil {
		t.Fatalf("Models failed: %v", err)
	}
	if len(resp.Models) != 1 || resp.Models[0].Quant != "Q3_K_XL" {
		t.Fatalf("unexpected models payload: %+v", resp.Models)
	}
}

func TestClientLoad(t *testing.T) {
	var seenBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/load" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		seenBody = body
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(LoadResponse{
			Status: "ok",
			Model:  "gemma-4-q3.gguf",
			Port:   11435,
			Device: "cuda:0",
			PID:    1234,
		})
	}))
	defer srv.Close()

	gpu := 0
	client := NewClient(srv.URL)
	resp, err := client.Load(LoadRequest{
		Model: "gemma-4-q3.gguf",
		GPU:   &gpu,
		CPU:   false,
	})
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if resp.Model != "gemma-4-q3.gguf" || resp.PID != 1234 {
		t.Fatalf("unexpected load response: %+v", resp)
	}

	var req LoadRequest
	if err := json.Unmarshal(seenBody, &req); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	if req.Model != "gemma-4-q3.gguf" || req.GPU == nil || *req.GPU != 0 || req.CPU {
		t.Fatalf("unexpected request payload: %+v", req)
	}
}

func TestClientUnload(t *testing.T) {
	var seenBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/unload" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		seenBody = body
		_ = json.NewEncoder(w).Encode(UnloadResponse{Status: "ok"})
	}))
	defer srv.Close()

	client := NewClient(srv.URL)
	if err := client.Unload("gemma-4-q3.gguf"); err != nil {
		t.Fatalf("Unload failed: %v", err)
	}

	var req UnloadRequest
	if err := json.Unmarshal(seenBody, &req); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	if req.Model != "gemma-4-q3.gguf" {
		t.Fatalf("unexpected unload request: %+v", req)
	}
}

func TestClientHealthOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/health" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := NewClient(srv.URL)
	if err := client.Health(); err != nil {
		t.Fatalf("Health failed: %v", err)
	}
}

func TestClientHealthDown(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`boom`))
	}))
	defer srv.Close()

	client := NewClient(srv.URL)
	if err := client.Health(); err == nil {
		t.Fatal("expected health error")
	}
}

func TestClientCheckModelExists(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/api/models" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(ModelsResponse{
			Models: []ModelInfo{
				{Name: "gpu-host.gguf", SizeBytes: 42},
			},
		})
	}))
	defer srv.Close()

	client := NewClient(srv.URL)
	exists, size, err := client.CheckModelExists("gpu-host.gguf")
	if err != nil {
		t.Fatalf("CheckModelExists failed: %v", err)
	}
	if !exists || size != 42 {
		t.Fatalf("unexpected result: exists=%v size=%d", exists, size)
	}
}

func TestClientUploadModel(t *testing.T) {
	var gotFilename string
	var gotSHA string
	var gotLength int64
	var gotBody []byte

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/upload" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		gotFilename = r.Header.Get("X-Filename")
		gotSHA = r.Header.Get("X-SHA256")
		gotLength = r.ContentLength
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		gotBody = body
		_ = json.NewEncoder(w).Encode(map[string]any{
			"filename": gotFilename,
			"size":     len(body),
			"sha256":   "beefcafe",
		})
	}))
	defer srv.Close()

	client := NewClient(srv.URL)
	sha, err := client.UploadModel("gpu-host.gguf", bytes.NewBufferString("abc"), 3, "deadbeef")
	if err != nil {
		t.Fatalf("UploadModel failed: %v", err)
	}
	if sha != "beefcafe" {
		t.Fatalf("sha256 = %q, want beefcafe", sha)
	}

	if gotFilename != "gpu-host.gguf" {
		t.Fatalf("X-Filename = %q, want gpu-host.gguf", gotFilename)
	}
	if gotSHA != "deadbeef" {
		t.Fatalf("X-SHA256 = %q, want deadbeef", gotSHA)
	}
	if gotLength != 3 {
		t.Fatalf("ContentLength = %d, want 3", gotLength)
	}
	if string(gotBody) != "abc" {
		t.Fatalf("body = %q, want abc", gotBody)
	}
}

func TestClientPull(t *testing.T) {
	var seenBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/pull" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		seenBody = body
		_ = json.NewEncoder(w).Encode(PullResponse{
			Filename: "Qwen3.6-35B-A3B-GGUF-Q4_K_S.gguf",
			Size:     12345,
		})
	}))
	defer srv.Close()

	client := NewClient(srv.URL)
	resp, err := client.Pull("unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S")
	if err != nil {
		t.Fatalf("Pull failed: %v", err)
	}
	if resp.Filename != "Qwen3.6-35B-A3B-GGUF-Q4_K_S.gguf" || resp.Size != 12345 {
		t.Fatalf("unexpected pull response: %+v", resp)
	}

	var req struct {
		Spec string `json:"spec"`
	}
	if err := json.Unmarshal(seenBody, &req); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	if req.Spec != "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S" {
		t.Fatalf("spec = %q, want pull spec", req.Spec)
	}
}

func TestClientPullError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "invalid spec"})
	}))
	defer srv.Close()

	client := NewClient(srv.URL)
	if _, err := client.Pull("bad-spec"); err == nil {
		t.Fatal("expected pull error")
	}
}
