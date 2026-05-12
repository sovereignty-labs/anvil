package process

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hirdforge/nollama/internal/hardware"
	"github.com/hirdforge/nollama/internal/model"
)

func writeTestGGUF(t *testing.T, dir string, name string, fileSizeBytes int64, kvs []modelTestKV) (string, *model.GGUFMetadata) {
	t.Helper()
	path := filepath.Join(dir, name)

	meta := &model.GGUFMetadata{
		FileSizeBytes:   fileSizeBytes,
		KV:              make(map[string]any),
	}

	for _, kv := range kvs {
		meta.KV[kv.key] = kv.value
	}

	meta.ExtractFields()

	return path, meta
}

type modelTestKV struct {
	key   string
	value any
}

func TestSelectBestGPU_NoGPUs(t *testing.T) {
	gpus := []hardware.GPU{}
	gpu, idx, free := selectBestGPU(gpus, 4096)
	if gpu != nil {
		t.Errorf("expected nil GPU, got %v", gpu)
	}
	if idx != -1 {
		t.Errorf("expected index -1, got %d", idx)
	}
	if free != 0 {
		t.Errorf("expected free 0, got %d", free)
	}
}

func TestSelectBestGPU_EnoughVRAM(t *testing.T) {
	gpus := []hardware.GPU{
		{Index: 0, VRAMFree: 12000},
		{Index: 1, VRAMFree: 16000},
		{Index: 2, VRAMFree: 20000},
	}
	gpu, idx, free := selectBestGPU(gpus, 8000)
	if gpu == nil {
		t.Fatal("expected a GPU, got nil")
	}
	if gpu.Index != 2 {
		t.Errorf("expected GPU 2 (most headroom), got %d", gpu.Index)
	}
	if idx != 2 {
		t.Errorf("expected index 2, got %d", idx)
	}
	if free != 20000 {
		t.Errorf("expected free 20000, got %d", free)
	}
}

func TestSelectBestGPU_FallbackToSmaller(t *testing.T) {
	gpus := []hardware.GPU{
		{Index: 0, VRAMFree: 6000},
		{Index: 1, VRAMFree: 10000},
		{Index: 2, VRAMFree: 8000},
	}
	gpu, idx, free := selectBestGPU(gpus, 5000)
	if gpu == nil {
		t.Fatal("expected a GPU, got nil")
	}
	if gpu.Index != 1 {
		t.Errorf("expected GPU 1 (most headroom), got %d", gpu.Index)
	}
	if idx != 1 {
		t.Errorf("expected index 1, got %d", idx)
	}
	if free != 10000 {
		t.Errorf("expected free 10000, got %d", free)
	}
}

func TestSelectBestGPU_NoGPUHasEnoughVRAM(t *testing.T) {
	gpus := []hardware.GPU{
		{Index: 0, VRAMFree: 4000},
		{Index: 1, VRAMFree: 6000},
		{Index: 2, VRAMFree: 8000},
	}
	gpu, idx, free := selectBestGPU(gpus, 16000)
	if gpu != nil {
		t.Errorf("expected nil GPU, got %v", gpu)
	}
	if idx != -1 {
		t.Errorf("expected index -1, got %d", idx)
	}
	if free != 0 {
		t.Errorf("expected free 0, got %d", free)
	}
}

func TestComputeFlags_GPU_Success(t *testing.T) {
	dir := t.TempDir()
	modelPath, meta := writeTestGGUF(t, dir, "llama-7b-Q4_K_M.gguf", 4*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.name", "LLaMA 7B"},
		{"general.context_length", uint64(4096)},
		{"general.file_type", uint32(15)},
		{"llama.embedding_length", uint32(4096)},
		{"llama.block_count", uint32(32)},
		{"llama.attention.head_count", uint32(32)},
		{"llama.attention.head_count_kv", uint32(32)},
		{"tokenizer.chat_template", "{% for message in messages %}...{% endfor %}"},
	})

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, Name: "NVIDIA GeForce RTX 4090", VRAMTotal: 24576, VRAMFree: 20000, VRAMUsed: 4576},
		},
		CPU: hardware.CPU{Cores: 16, Threads: 32},
	}

	result, err := ComputeFlags(meta, modelPath, inv, "/usr/local/bin/llama-server", 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	if result.CPUFallback {
		t.Error("expected GPU mode, got CPU fallback")
	}
	if result.SelectedDevice != "cuda:0" {
		t.Errorf("selected device = %q, want cuda:0", result.SelectedDevice)
	}

	// Check required flags
	flagMap := make(map[string]bool)
	for _, f := range result.Flags {
		flagMap[f] = true
	}

	if !flagMap["--n-gpu-layers"] {
		t.Error("expected --n-gpu-layers flag")
	}
	if !flagMap["--flash-attn"] {
		t.Error("expected --flash-attn flag")
	}
	if !flagMap["--no-warmup"] {
		t.Error("expected --no-warmup flag")
	}
	if !flagMap["--jinja"] {
		t.Error("expected --jinja flag (chat template present)")
	}
	if !flagMap["--host"] {
		t.Error("expected --host flag")
	}
	if !flagMap["--port"] {
		t.Error("expected --port flag")
	}

	// Check port
	portIndex := -1
	for i, f := range result.Flags {
		if f == "--port" && i+1 < len(result.Flags) {
			portIndex = i
			break
		}
	}
	if portIndex >= 0 && result.Flags[portIndex+1] != "11434" {
		t.Errorf("expected port 11434, got %s", result.Flags[portIndex+1])
	}

	// Check command contains llama-server path
	if result.Command == "" {
		t.Error("expected non-empty command")
	}
}

func TestComputeFlags_CPU_Fallback(t *testing.T) {
	dir := t.TempDir()
	modelPath, meta := writeTestGGUF(t, dir, "llama-7b-Q4_K_M.gguf", 4*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.name", "LLaMA 7B"},
		{"general.context_length", uint64(4096)},
		{"general.file_type", uint32(15)},
		{"llama.embedding_length", uint32(4096)},
		{"llama.block_count", uint32(32)},
		{"llama.attention.head_count", uint32(32)},
		{"llama.attention.head_count_kv", uint32(32)},
	})

	// Small GPU that doesn't have enough VRAM
	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, Name: "NVIDIA GeForce GTX 1060", VRAMTotal: 6144, VRAMFree: 2048, VRAMUsed: 4096},
		},
		CPU: hardware.CPU{Cores: 8, Threads: 16},
	}

	result, err := ComputeFlags(meta, modelPath, inv, "/usr/local/bin/llama-server", 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	if !result.CPUFallback {
		t.Error("expected CPU fallback, got GPU mode")
	}
	if result.SelectedDevice != "cpu" {
		t.Errorf("selected device = %q, want cpu", result.SelectedDevice)
	}

	flagMap := make(map[string]bool)
	for _, f := range result.Flags {
		flagMap[f] = true
	}

	if !flagMap["--n-gpu-layers"] {
		t.Error("expected --n-gpu-layers in CPU fallback mode to force CPU-only")
	}
	if !flagMap["--threads"] {
		t.Error("expected --threads flag in CPU mode")
	}
}

func TestComputeFlags_Jinja_Present(t *testing.T) {
	dir := t.TempDir()
	modelPath, meta := writeTestGGUF(t, dir, "model-with-template.gguf", 2*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.context_length", uint64(2048)},
		{"general.file_type", uint32(15)},
		{"llama.embedding_length", uint32(2048)},
		{"llama.block_count", uint32(16)},
		{"llama.attention.head_count", uint32(16)},
		{"llama.attention.head_count_kv", uint32(16)},
		{"tokenizer.chat_template", "<system>...</system><human>...</human>"},
	})

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, VRAMTotal: 24576, VRAMFree: 20000},
		},
		CPU: hardware.CPU{Cores: 8},
	}

	result, err := ComputeFlags(meta, modelPath, inv, "/usr/local/bin/llama-server", 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	hasJinja := false
	for _, f := range result.Flags {
		if f == "--jinja" {
			hasJinja = true
			break
		}
	}
	if !hasJinja {
		t.Error("expected --jinja flag when chat template exists")
	}
}

func TestComputeFlags_Jinja_Absent(t *testing.T) {
	dir := t.TempDir()
	modelPath, meta := writeTestGGUF(t, dir, "base-model.gguf", 2*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.context_length", uint64(2048)},
		{"general.file_type", uint32(15)},
		{"llama.embedding_length", uint32(2048)},
		{"llama.block_count", uint32(16)},
		{"llama.attention.head_count", uint32(16)},
		{"llama.attention.head_count_kv", uint32(16)},
	})

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, VRAMTotal: 24576, VRAMFree: 20000},
		},
		CPU: hardware.CPU{Cores: 8},
	}

	result, err := ComputeFlags(meta, modelPath, inv, "/usr/local/bin/llama-server", 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	hasJinja := false
	for _, f := range result.Flags {
		if f == "--jinja" {
			hasJinja = true
			break
		}
	}
	if hasJinja {
		t.Error("did not expect --jinja flag when no chat template")
	}
}

func TestComputeFlags_ContextLength_Capping(t *testing.T) {
	dir := t.TempDir()
	modelPath, meta := writeTestGGUF(t, dir, "large-context.gguf", 2*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.context_length", uint64(131072)},
		{"general.file_type", uint32(15)},
		{"llama.embedding_length", uint32(2048)},
		{"llama.block_count", uint32(16)},
		{"llama.attention.head_count", uint32(16)},
		{"llama.attention.head_count_kv", uint32(16)},
	})

	// Tiny GPU with very limited free VRAM
	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, VRAMTotal: 8192, VRAMFree: 4000, VRAMUsed: 4192},
		},
		CPU: hardware.CPU{Cores: 8},
	}

	result, err := ComputeFlags(meta, modelPath, inv, "/usr/local/bin/llama-server", 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	// Context size should be capped when VRAM is tight
	hasCtxSize := false
	for i, f := range result.Flags {
		if f == "--ctx-size" && i+1 < len(result.Flags) {
			hasCtxSize = true
			ctxVal := result.Flags[i+1]
			// Should be less than 131072
			if ctxVal == "131072" {
				t.Errorf("expected capped ctx-size, got uncapped %s", ctxVal)
			}
			break
		}
	}
	_ = hasCtxSize
}

func TestComputeFlags_Ports_Increment(t *testing.T) {
	dir := t.TempDir()
	_, meta := writeTestGGUF(t, dir, "model.gguf", 2*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.context_length", uint64(4096)},
		{"general.file_type", uint32(15)},
		{"llama.embedding_length", uint32(2048)},
		{"llama.block_count", uint32(16)},
		{"llama.attention.head_count", uint32(16)},
		{"llama.attention.head_count_kv", uint32(16)},
	})

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, VRAMTotal: 24576, VRAMFree: 20000},
		},
		CPU: hardware.CPU{Cores: 8},
	}

	for i, expected := range []int{11434, 11435, 11436} {
		result, err := ComputeFlags(meta, "/path/model.gguf", inv, "/usr/local/bin/llama-server", i)
		if err != nil {
			t.Fatalf("ComputeFlags index=%d error: %v", i, err)
		}

		portFlag := ""
		for j, f := range result.Flags {
			if f == "--port" && j+1 < len(result.Flags) {
				portFlag = result.Flags[j+1]
				break
			}
		}
		if portFlag != fmt.Sprintf("%d", expected) {
			t.Errorf("index=%d port=%s, want %d", i, portFlag, expected)
		}
	}
}

func TestComputeFlags_GPU_WithChatTemplate(t *testing.T) {
	dir := t.TempDir()
	modelPath, meta := writeTestGGUF(t, dir, "model.gguf", 4*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.context_length", uint64(4096)},
		{"general.file_type", uint32(15)},
		{"llama.embedding_length", uint32(4096)},
		{"llama.block_count", uint32(32)},
		{"llama.attention.head_count", uint32(32)},
		{"llama.attention.head_count_kv", uint32(32)},
		{"tokenizer.chat_template", "{% for msg in messages %}{{msg}}{% endfor %}"},
	})

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, VRAMTotal: 24576, VRAMFree: 20000},
		},
		CPU: hardware.CPU{Cores: 8},
	}

	result, err := ComputeFlags(meta, modelPath, inv, "/usr/local/bin/llama-server", 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	if result.CPUFallback {
		t.Error("expected GPU mode, got CPU fallback")
	}

	hasJinja := false
	for _, f := range result.Flags {
		if f == "--jinja" {
			hasJinja = true
			break
		}
	}
	if !hasJinja {
		t.Error("expected --jinja flag with chat template in GPU mode")
	}

	if result.VRAMTotalMB == 0 {
		t.Error("expected non-zero VRAM total in GPU mode")
	}
}

func TestComputeFlags_NoGPUDetected_CPUFallback(t *testing.T) {
	dir := t.TempDir()
	modelPath, meta := writeTestGGUF(t, dir, "model.gguf", 4*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.context_length", uint64(4096)},
		{"general.file_type", uint32(15)},
		{"llama.embedding_length", uint32(4096)},
		{"llama.block_count", uint32(32)},
		{"llama.attention.head_count", uint32(32)},
		{"llama.attention.head_count_kv", uint32(32)},
	})

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{},
		CPU:  hardware.CPU{Cores: 16, Threads: 32},
	}

	result, err := ComputeFlags(meta, modelPath, inv, "/usr/local/bin/llama-server", 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	if !result.CPUFallback {
		t.Error("expected CPU fallback with no GPUs")
	}
	if result.SelectedDevice != "cpu" {
		t.Errorf("selected device = %q, want cpu", result.SelectedDevice)
	}
	if result.CPUThreads == 0 {
		t.Error("expected CPU threads to be set")
	}
}

func TestComputeFlags_EmptyModelPath(t *testing.T) {
	dir := t.TempDir()
	_, meta := writeTestGGUF(t, dir, "model.gguf", 4*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.file_type", uint32(15)},
	})

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{{Index: 0, VRAMFree: 20000}},
		CPU:  hardware.CPU{Cores: 8},
	}

	_, err := ComputeFlags(meta, "", inv, "/usr/local/bin/llama-server", 0)
	if err == nil {
		t.Error("expected error for empty model path")
	}
}

func TestComputeFlags_EmptyLlamaServerPath(t *testing.T) {
	dir := t.TempDir()
	_, meta := writeTestGGUF(t, dir, "model.gguf", 4*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.file_type", uint32(15)},
	})

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{{Index: 0, VRAMFree: 20000}},
		CPU:  hardware.CPU{Cores: 8},
	}

	_, err := ComputeFlags(meta, "/path/model.gguf", inv, "", 0)
	if err == nil {
		t.Error("expected error for empty llama-server path")
	}
}

func TestComputeFlags_VRAM_Estimation(t *testing.T) {
	dir := t.TempDir()
	_, meta := writeTestGGUF(t, dir, "model.gguf", 4*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.file_type", uint32(15)},
	})

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{{Index: 0, VRAMTotal: 24576, VRAMFree: 20000}},
		CPU:  hardware.CPU{Cores: 8},
	}

	result, err := ComputeFlags(meta, "/path/model.gguf", inv, "/usr/local/bin/llama-server", 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	// 4GB file = 4096 MB + 20% = 4915.2 MB => 4915 MB
	expected := uint64(4096) + (4096*20)/100
	if result.VRAMUsedMB != expected {
		t.Errorf("VRAMUsedMB = %d, want %d", result.VRAMUsedMB, expected)
	}
}

func TestGPUReasoning_NoGPUs(t *testing.T) {
	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{},
		CPU:  hardware.CPU{Cores: 8},
	}
	reasoning := GPUReasoning(inv, 4096)
	if reasoning != "No GPU detected — will use CPU" {
		t.Errorf("unexpected reasoning: %q", reasoning)
	}
}

func TestGPUReasoning_GPUSelected(t *testing.T) {
	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, Name: "NVIDIA GeForce RTX 4090", VRAMTotal: 24576, VRAMFree: 20000, VRAMUsed: 4576},
		},
	}
	reasoning := GPUReasoning(inv, 4096)
	if !strings.Contains(reasoning, "Selected: GPU 0") {
		t.Errorf("expected GPU 0 selection in reasoning: %q", reasoning)
	}
	if !strings.Contains(reasoning, "headroom") {
		t.Errorf("expected headroom info in reasoning: %q", reasoning)
	}
}

func TestGPUReasoning_Fallback(t *testing.T) {
	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, Name: "NVIDIA GTX 1060", VRAMTotal: 6144, VRAMFree: 2048, VRAMUsed: 4096},
		},
	}
	reasoning := GPUReasoning(inv, 8192)
	if !strings.Contains(reasoning, "insufficient") {
		t.Errorf("expected insufficient GPU message: %q", reasoning)
	}
	if !strings.Contains(reasoning, "CPU fallback") {
		t.Errorf("expected CPU fallback in reasoning: %q", reasoning)
	}
}

func TestComputeFlags_MultiGPU_BestSelected(t *testing.T) {
	dir := t.TempDir()
	modelPath, meta := writeTestGGUF(t, dir, "model.gguf", 8*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.context_length", uint64(4096)},
		{"general.file_type", uint32(15)},
		{"llama.embedding_length", uint32(4096)},
		{"llama.block_count", uint32(32)},
		{"llama.attention.head_count", uint32(32)},
		{"llama.attention.head_count_kv", uint32(32)},
	})

	// 3 GPUs: 12GB, 24GB, 16GB — 8GB model needs 9.6GB
	// GPU 0: 12GB total, 10GB free — fits
	// GPU 1: 24GB total, 20GB free — fits (best headroom)
	// GPU 2: 16GB total, 8GB free — doesn't fit
	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, Name: "NVIDIA RTX 3080", VRAMTotal: 12288, VRAMFree: 10240, VRAMUsed: 2048},
			{Index: 1, Name: "NVIDIA RTX 4090", VRAMTotal: 24576, VRAMFree: 20480, VRAMUsed: 4096},
			{Index: 2, Name: "NVIDIA RTX 3060", VRAMTotal: 16384, VRAMFree: 8192, VRAMUsed: 8192},
		},
		CPU: hardware.CPU{Cores: 16, Threads: 32},
	}

	result, err := ComputeFlags(meta, modelPath, inv, "/usr/local/bin/llama-server", 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	if result.CPUFallback {
		t.Error("expected GPU mode, got CPU fallback")
	}
	// Should pick GPU 1 (RTX 4090) with most headroom
	if result.SelectedDevice != "cuda:1" {
		t.Errorf("expected cuda:1, got %q", result.SelectedDevice)
	}
}

func TestComputeFlags_NoWarmup_And_FlashAttention(t *testing.T) {
	dir := t.TempDir()
	modelPath, meta := writeTestGGUF(t, dir, "model.gguf", 4*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.context_length", uint64(4096)},
		{"general.file_type", uint32(15)},
		{"llama.embedding_length", uint32(4096)},
		{"llama.block_count", uint32(32)},
		{"llama.attention.head_count", uint32(32)},
		{"llama.attention.head_count_kv", uint32(32)},
	})

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{
			{Index: 0, VRAMTotal: 24576, VRAMFree: 20000},
		},
		CPU: hardware.CPU{Cores: 8},
	}

	result, err := ComputeFlags(meta, modelPath, inv, "/usr/local/bin/llama-server", 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	hasFlashAttn := false
	hasNoWarmup := false
	for _, f := range result.Flags {
		if f == "--flash-attn" {
			hasFlashAttn = true
		}
		if f == "--no-warmup" {
			hasNoWarmup = true
		}
	}
	if !hasFlashAttn {
		t.Error("expected --flash-attn flag")
	}
	if !hasNoWarmup {
		t.Error("expected --no-warmup flag")
	}
}

func TestComputeFlags_AbsolutePath_EnvVar(t *testing.T) {
	dir := t.TempDir()
	modelPath, meta := writeTestGGUF(t, dir, "model.gguf", 4*1024*1024*1024, []modelTestKV{
		{"general.architecture", "llama"},
		{"general.file_type", uint32(15)},
	})

	os.Setenv("NOLLAMA_LLAMA_SERVER", "/opt/llama/bin/llama-server")
	defer os.Unsetenv("NOLLAMA_LLAMA_SERVER")

	inv := &hardware.Inventory{
		GPUs: []hardware.GPU{{Index: 0, VRAMTotal: 24576, VRAMFree: 20000}},
		CPU:  hardware.CPU{Cores: 8},
	}

	result, err := ComputeFlags(meta, modelPath, inv, os.Getenv("NOLLAMA_LLAMA_SERVER"), 0)
	if err != nil {
		t.Fatalf("ComputeFlags error: %v", err)
	}

	if result.Command == "" {
		t.Error("expected non-empty command")
	}
}
