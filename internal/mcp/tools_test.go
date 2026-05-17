package mcp

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hirdforge/nollama/internal/config"
	"github.com/hirdforge/nollama/internal/model"
	mcpkit "github.com/mark3labs/mcp-go/mcp"
)

func TestToolStatusAggregatesLocalAndRemote(t *testing.T) {
	localSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/status" {
			t.Fatalf("unexpected local path: %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(mcpStatusResponse("local-model.gguf", "RTX 4090", 20*1024, 24*1024, 1111, 3600))
	}))
	defer localSrv.Close()

	remoteSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/status" {
			t.Fatalf("unexpected remote path: %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(mcpStatusResponse("gpu-host-model.gguf", "RTX 3090", 10*1024, 24*1024, 2222, 7200))
	}))
	defer remoteSrv.Close()

	localURL, _ := url.Parse(localSrv.URL)
	cfg := config.DefaultConfig()
	cfg.Bind = localURL.Host
	cfg.Remotes = map[string]config.Remote{
		"gpu-host": {URL: remoteSrv.URL},
	}

	runner := NewRunner(cfg, filepath.Join(t.TempDir(), "remotes.yaml"))
	statusRes, err := runner.toolStatus(context.Background(), callTool("nollama_status", nil))
	text := mustToolText(t, statusRes, err)
	if !strings.Contains(text, "local") || !strings.Contains(text, "gpu-host") {
		t.Fatalf("status output missing node labels:\n%s", text)
	}
	if !strings.Contains(text, "local-model.gguf") || !strings.Contains(text, "gpu-host-model.gguf") {
		t.Fatalf("status output missing model names:\n%s", text)
	}
}

func TestToolLoadUnloadModelsPullAndRm(t *testing.T) {
	dir := t.TempDir()
	_ = writeTestGGUF(t, dir, "local-Q8_K_XL.gguf", []testKV{
		{"general.architecture", model.GGUFTypeString, "llama"},
		{"general.context_length", model.GGUFTypeUint64, uint64(8192)},
		{"general.file_type", model.GGUFTypeUint32, uint32(7)},
	})

	modelSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/load":
			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatal(err)
			}
			if req["model"] != "gpu-host.gguf" {
				t.Fatalf("load model = %v", req["model"])
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"status": "ok",
				"model":  "gpu-host.gguf",
				"port":   11435,
				"device": "cuda:0",
				"pid":    1234,
			})
		case "/api/unload":
			_ = json.NewEncoder(w).Encode(map[string]any{"status": "ok"})
		case "/api/models":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"models": []map[string]any{{
					"name":           "gpu-host.gguf",
					"size_bytes":     1024,
					"size_human":     "1 KB",
					"arch":           "llama",
					"quant":          "Q4_K_S",
					"context_length": 4096,
				}},
			})
		case "/api/pull":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"filename": "gpu-host.gguf",
				"size":     123,
			})
		case "/api/rm":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"filename": "gpu-host.gguf",
				"deleted":  true,
			})
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer modelSrv.Close()

	localURL, _ := url.Parse(modelSrv.URL)
	cfg := config.DefaultConfig()
	cfg.Bind = localURL.Host
	cfg.ModelDir = dir

	runner := NewRunner(cfg, filepath.Join(t.TempDir(), "remotes.yaml"))

	loadRes, err := runner.toolLoad(context.Background(), callTool("nollama_load", map[string]any{
		"model": "gpu-host.gguf",
		"gpu":   0,
		"flags": map[string]any{"ctx-size": 8192},
	}))
	if err != nil {
		t.Fatal(err)
	}
	assertTextContains(t, loadRes, "Loaded model gpu-host.gguf")

	unloadRes, err := runner.toolUnload(context.Background(), callTool("nollama_unload", map[string]any{
		"model": "gpu-host.gguf",
	}))
	if err != nil {
		t.Fatal(err)
	}
	assertTextContains(t, unloadRes, "Unloaded model gpu-host.gguf")

	modelsRes, err := runner.toolModels(context.Background(), callTool("nollama_models", nil))
	if err != nil {
		t.Fatal(err)
	}
	assertTextContains(t, modelsRes, "MODEL")
	assertTextContains(t, modelsRes, "local-Q8_K_XL")

	pullRes, err := runner.toolPull(context.Background(), callTool("nollama_pull", map[string]any{
		"spec": "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S",
	}))
	if err != nil {
		t.Fatal(err)
	}
	assertTextContains(t, pullRes, "Pulled gpu-host.gguf")

	rmRes, err := runner.toolRm(context.Background(), callTool("nollama_rm", map[string]any{
		"model": "gpu-host",
	}))
	if err != nil {
		t.Fatal(err)
	}
	assertTextContains(t, rmRes, "Removed gpu-host")
}

func TestToolInspectAndRuntimes(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir, "inspect.gguf", []testKV{
		{"general.architecture", model.GGUFTypeString, "llama"},
		{"general.name", model.GGUFTypeString, "Inspect Model"},
		{"general.context_length", model.GGUFTypeUint64, uint64(4096)},
		{"general.file_type", model.GGUFTypeUint32, uint32(15)},
	})

	cfg := config.DefaultConfig()
	cfg.ModelDir = dir
	runner := NewRunner(cfg, filepath.Join(t.TempDir(), "remotes.yaml"))

	inspectRes, err := runner.toolInspect(context.Background(), callTool("nollama_inspect", map[string]any{
		"model": filepath.Base(path),
	}))
	if err != nil {
		t.Fatal(err)
	}
	assertTextContains(t, inspectRes, "Model:")
	assertTextContains(t, inspectRes, "Arch:")

	runtimesRes, err := runner.toolRuntimes(context.Background(), callTool("nollama_runtimes", nil))
	if err != nil {
		t.Fatal(err)
	}
	if len(runtimesRes.Content) == 0 {
		t.Fatal("expected runtimes text")
	}
}

func TestToolValidation(t *testing.T) {
	runner := NewRunner(config.DefaultConfig(), filepath.Join(t.TempDir(), "remotes.yaml"))

	cases := []struct {
		name string
		fn   func(context.Context, mcpkit.CallToolRequest) (*mcpkit.CallToolResult, error)
		req  mcpkit.CallToolRequest
	}{
		{
			name: "load",
			fn:   runner.toolLoad,
			req:  callTool("nollama_load", nil),
		},
		{
			name: "pull",
			fn:   runner.toolPull,
			req:  callTool("nollama_pull", nil),
		},
		{
			name: "rm",
			fn:   runner.toolRm,
			req:  callTool("nollama_rm", nil),
		},
		{
			name: "inspect",
			fn:   runner.toolInspect,
			req:  callTool("nollama_inspect", nil),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := tc.fn(context.Background(), tc.req)
			if err != nil {
				t.Fatal(err)
			}
			if res == nil || !res.IsError {
				t.Fatalf("expected error result for %s", tc.name)
			}
		})
	}
}

func callTool(name string, args map[string]any) mcpkit.CallToolRequest {
	if args == nil {
		args = map[string]any{}
	}
	return mcpkit.CallToolRequest{
		Params: mcpkit.CallToolParams{
			Name:      name,
			Arguments: args,
		},
	}
}

func mustToolText(t *testing.T, res *mcpkit.CallToolResult, err error) string {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	assertTextContains(t, res, "")
	return res.Content[0].(mcpkit.TextContent).Text
}

func assertTextContains(t *testing.T, res *mcpkit.CallToolResult, want string) {
	t.Helper()
	if res == nil {
		t.Fatal("nil tool result")
	}
	if len(res.Content) == 0 {
		t.Fatal("tool result has no content")
	}
	text := res.Content[0].(mcpkit.TextContent).Text
	if want != "" && !strings.Contains(text, want) {
		t.Fatalf("tool text missing %q:\n%s", want, text)
	}
}

func mcpStatusResponse(modelName, gpuName string, free, total uint64, pid int, uptime int64) map[string]any {
	return map[string]any{
		"models": []map[string]any{{
			"name":           modelName,
			"port":           11435,
			"gpu":            "cuda:0",
			"pid":            pid,
			"uptime_seconds": uptime,
		}},
		"node": map[string]any{
			"gpus": []map[string]any{{
				"index":         0,
				"name":          gpuName,
				"vram_total_mb": total,
				"vram_free_mb":  free,
			}},
			"ram_total_mb": 32768,
			"ram_free_mb":  16384,
		},
	}
}

type testKV struct {
	key   string
	vtype uint32
	value any
}

func writeTestGGUF(t *testing.T, dir, name string, kvs []testKV) string {
	t.Helper()
	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w := binary.LittleEndian
	_, _ = f.Write([]byte("GGUF"))
	_ = binary.Write(f, w, uint32(3))
	_ = binary.Write(f, w, uint64(0))
	_ = binary.Write(f, w, uint64(len(kvs)))
	for _, kv := range kvs {
		_ = binary.Write(f, w, uint64(len(kv.key)))
		_, _ = f.Write([]byte(kv.key))
		_ = binary.Write(f, w, kv.vtype)
		switch kv.vtype {
		case model.GGUFTypeString:
			s := kv.value.(string)
			_ = binary.Write(f, w, uint64(len(s)))
			_, _ = f.Write([]byte(s))
		case model.GGUFTypeUint32:
			_ = binary.Write(f, w, kv.value.(uint32))
		case model.GGUFTypeUint64:
			_ = binary.Write(f, w, kv.value.(uint64))
		default:
			t.Fatalf("unsupported type %d", kv.vtype)
		}
	}
	return path
}
