// Package process computes llama-server flags from GGUF metadata and hardware inventory.
package process

import (
	"fmt"
	"strings"
	"time"

	"github.com/sovereignty-labs/anvil/internal/hardware"
	"github.com/sovereignty-labs/anvil/internal/model"
	runtimemgr "github.com/sovereignty-labs/anvil/internal/runtime"
)

// Result holds the computed flags, device selection, and metadata for a model load.
type Result struct {
	SelectedDevice string // "cuda:0", "vulkan:1", "cpu", etc.
	Backend        runtimemgr.BuildBackend
	Flags          []string // llama-server flags as []string
	Command        string   // full command line string
	VRAMUsedMB     uint64   // estimated VRAM usage in MiB
	VRAMTotalMB    uint64   // available VRAM on selected device in MiB
	CPUFallback    bool     // whether CPU fallback was used
	CPUThreads     int      // number of CPU threads (non-zero only when CPU fallback)
	GPUIndex       int      // GPU index used (-1 for CPU fallback)
	Port           int      // llama-server HTTP port (11434 + modelIndex)
	ReadyTimeout   time.Duration
}

// EnsureHostFlag injects --host 0.0.0.0 into flags when no --host is already
// present. llama-server itself defaults to 127.0.0.1, which makes spawned
// instances unreachable from federation peers or other machines on the LAN —
// the inverse of what anvil is for. Called on the final merged flag slice
// (after smart defaults + config defaults + profiles + passthrough) so any
// explicit user-supplied --host wins.
func EnsureHostFlag(flags []string) []string {
	for _, f := range flags {
		if f == "--host" {
			return flags
		}
	}
	return append(flags, "--host", "0.0.0.0")
}

// OverrideResultPort rewrites the port baked into a ComputeFlags result so the
// process is spawned on a pinned port. No-op when port <= 0. Updates both the
// Result.Port field and the --port value inside Result.Flags.
func OverrideResultPort(result *Result, port int) {
	if result == nil || port <= 0 {
		return
	}
	result.Port = port
	for i := 0; i < len(result.Flags); i++ {
		if result.Flags[i] == "--port" && i+1 < len(result.Flags) {
			result.Flags[i+1] = fmt.Sprintf("%d", port)
			return
		}
	}
	result.Flags = append(result.Flags, "--port", fmt.Sprintf("%d", port))
}

// ComputeFlags takes a GGUF model (with its path), hardware inventory and returns the
// optimal llama-server flags for loading the model.
func ComputeFlags(meta *model.GGUFMetadata, modelPath string, inv *hardware.Inventory, llamaServerPath string, modelIndex int, backend ...runtimemgr.BuildBackend) (*Result, error) {
	if llamaServerPath == "" {
		return nil, fmt.Errorf("llama-server path is required")
	}
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}

	port := 11434 + modelIndex
	effectiveBackend := normalizeBackend(backend...)
	if effectiveBackend == runtimemgr.BuildBackendAuto {
		effectiveBackend = runtimemgr.BuildBackendCUDA
	}
	if inv == nil {
		inv = &hardware.Inventory{}
	}

	result := &Result{
		Backend: effectiveBackend,
		Flags: []string{
			"--model", modelPath,
			"--host", "0.0.0.0",
			"--port", fmt.Sprintf("%d", port),
		},
		Port: port,
	}

	// Estimate required VRAM: file size + 20% overhead for KV cache
	fileSizeMB := uint64(meta.FileSizeBytes) / 1024 / 1024
	requiredVRAM := fileSizeMB + (fileSizeMB*20)/100
	result.VRAMUsedMB = requiredVRAM

	var bestGPU *hardware.GPU
	var bestVulkan *hardware.VulkanGPU
	var availableVRAM uint64

	switch effectiveBackend {
	case runtimemgr.BuildBackendCPU:
		applyCPUFallback(result, meta, inv)
	case runtimemgr.BuildBackendVulkan:
		bestVulkan, _, availableVRAM = selectBestVulkanGPU(inv.VulkanGPUs, requiredVRAM)
	case runtimemgr.BuildBackendROCm:
		bestGPU, _, availableVRAM = selectBestGPU(inv.ROCmGPUs, requiredVRAM)
	default:
		bestGPU, _, availableVRAM = selectBestGPU(inv.GPUs, requiredVRAM)
	}

	if !result.CPUFallback && (bestGPU != nil || bestVulkan != nil) {
		result.Flags = append(result.Flags,
			"--flash-attn", "on",
			"--no-warmup",
		)
		if effectiveBackend != runtimemgr.BuildBackendVulkan {
			result.Flags = append(result.Flags, "--n-gpu-layers", "99")
		} else {
			result.Flags = append(result.Flags, "--n-gpu-layers", "-1")
		}
		result.VRAMTotalMB = availableVRAM
		if bestGPU != nil {
			result.SelectedDevice = fmt.Sprintf("%s:%d", effectiveBackend, bestGPU.Index)
			result.GPUIndex = bestGPU.Index
		} else {
			result.SelectedDevice = fmt.Sprintf("%s:%d", effectiveBackend, bestVulkan.Index)
			result.GPUIndex = bestVulkan.Index
		}

		// Context size: start from GGUF metadata, cap by available VRAM
		ctxSize := int(meta.ContextLength)
		if ctxSize > 0 {
			capped := capContextByVRAM(ctxSize, meta, availableVRAM)
			if capped < ctxSize {
				result.Flags = append(result.Flags,
					"--ctx-size", fmt.Sprintf("%d", capped),
				)
			}
		}

		// --jinja if the model has a chat template
		if meta.HasChatTemplate {
			result.Flags = append(result.Flags, "--jinja")
		}
	} else if !result.CPUFallback {
		applyCPUFallback(result, meta, inv)
	}

	// Build the full command line
	result.Command = buildCommand(llamaServerPath, result.Flags)

	return result, nil
}

func normalizeBackend(backend ...runtimemgr.BuildBackend) runtimemgr.BuildBackend {
	if len(backend) == 0 {
		return runtimemgr.BuildBackendCUDA
	}
	switch backend[0] {
	case runtimemgr.BuildBackendCUDA, runtimemgr.BuildBackendROCm, runtimemgr.BuildBackendVulkan, runtimemgr.BuildBackendCPU:
		return backend[0]
	default:
		return runtimemgr.BuildBackendCUDA
	}
}

func applyCPUFallback(result *Result, meta *model.GGUFMetadata, inv *hardware.Inventory) {
	result.CPUFallback = true
	result.SelectedDevice = "cpu"
	result.GPUIndex = -1
	result.Flags = append(result.Flags,
		"--n-gpu-layers", "0",
		"--no-warmup",
	)
	result.VRAMTotalMB = 0

	// Context size: still cap against available system RAM
	ctxSize := int(meta.ContextLength)
	if ctxSize > 0 {
		capped := capContextByVRAM(ctxSize, meta, inv.CPU.RAMFreeMB)
		if capped < ctxSize {
			result.Flags = append(result.Flags,
				"--ctx-size", fmt.Sprintf("%d", capped),
			)
		}
	}

	// --jinja if the model has a chat template
	if meta.HasChatTemplate {
		result.Flags = append(result.Flags, "--jinja")
	}

	// --threads based on CPU cores
	threads := inv.CPU.Cores
	if threads == 0 {
		threads = inv.CPU.Threads
	}
	if threads == 0 {
		threads = 4
	}
	result.CPUThreads = threads
	result.Flags = append(result.Flags,
		"--threads", fmt.Sprintf("%d", threads),
	)
}

// selectBestGPU picks the GPU with enough free VRAM, preferring the one with
// the most headroom (free VRAM - required). Returns (gpu, index, freeVRAM) or (nil, -1, 0).
func selectBestGPU(gpus []hardware.GPU, requiredVRAM uint64) (*hardware.GPU, int, uint64) {
	if len(gpus) == 0 {
		return nil, -1, 0
	}

	var best *hardware.GPU
	bestIdx := -1
	bestHeadroom := uint64(0)

	for i, g := range gpus {
		free := g.VRAMFree
		if free < requiredVRAM {
			continue
		}
		headroom := free - requiredVRAM
		if best == nil || headroom > bestHeadroom {
			best = &g
			bestIdx = i
			bestHeadroom = headroom
		}
	}

	if best == nil {
		return nil, -1, 0
	}
	return best, bestIdx, best.VRAMFree
}

func selectBestVulkanGPU(gpus []hardware.VulkanGPU, requiredVRAM uint64) (*hardware.VulkanGPU, int, uint64) {
	candidates := usableVulkanGPUs(gpus)
	if len(candidates) == 0 {
		return nil, -1, 0
	}

	var best *hardware.VulkanGPU
	bestIdx := -1
	bestHeadroom := uint64(0)

	for i, g := range candidates {
		free := g.FreeVRAM
		if free < requiredVRAM {
			continue
		}
		headroom := free - requiredVRAM
		if best == nil || headroom > bestHeadroom {
			best = &g
			bestIdx = i
			bestHeadroom = headroom
		}
	}

	if best == nil {
		return nil, -1, 0
	}
	return best, bestIdx, best.FreeVRAM
}

func usableVulkanGPUs(gpus []hardware.VulkanGPU) []hardware.VulkanGPU {
	usable := make([]hardware.VulkanGPU, 0, len(gpus))
	for _, g := range gpus {
		if vulkanGPUIsSoftware(g) {
			continue
		}
		usable = append(usable, g)
	}
	if len(usable) == 0 {
		return usable
	}

	hasAMDOrIntelDiscrete := false
	for _, g := range usable {
		if !vulkanGPUIsDiscrete(g) {
			continue
		}
		switch vulkanGPUVendor(g) {
		case "amd", "intel":
			hasAMDOrIntelDiscrete = true
			break
		}
	}
	if !hasAMDOrIntelDiscrete {
		return usable
	}

	filtered := make([]hardware.VulkanGPU, 0, len(usable))
	for _, g := range usable {
		if vulkanGPUVendor(g) == "nvidia" {
			continue
		}
		filtered = append(filtered, g)
	}
	return filtered
}

func vulkanGPUIsSoftware(g hardware.VulkanGPU) bool {
	name := strings.ToLower(g.Name)
	deviceType := strings.ToUpper(g.DeviceType())
	return strings.Contains(name, "llvmpipe") ||
		strings.Contains(name, "lavapipe") ||
		strings.Contains(name, "software") ||
		strings.Contains(deviceType, "CPU")
}

func vulkanGPUIsDiscrete(g hardware.VulkanGPU) bool {
	return strings.Contains(strings.ToUpper(g.DeviceType()), "DISCRETE_GPU")
}

func vulkanGPUVendor(g hardware.VulkanGPU) string {
	switch strings.ToLower(strings.TrimSpace(g.VendorID())) {
	case "0x10de":
		return "nvidia"
	case "0x1002":
		return "amd"
	case "0x8086":
		return "intel"
	}
	name := strings.ToLower(g.Name)
	switch {
	case strings.Contains(name, "nvidia"):
		return "nvidia"
	case strings.Contains(name, "amd"), strings.Contains(name, "radeon"):
		return "amd"
	case strings.Contains(name, "intel"), strings.Contains(name, "arc"):
		return "intel"
	default:
		return ""
	}
}

// capContextByVRAM estimates the maximum safe context length given available VRAM.
// Conservatively allocates ~2 bytes per context token for KV cache.
func capContextByVRAM(ideal int, meta *model.GGUFMetadata, availableMB uint64) int {
	if ideal <= 0 {
		return 4096
	}

	bytesPerToken := uint64(2)
	availableBytes := availableMB * 1024 * 1024

	// Reserve 50% of available VRAM for the model weights themselves
	reservedForModel := availableBytes / 2
	maxTokens := int(reservedForBytes(bytesPerToken, reservedForModel))

	if maxTokens < 512 {
		maxTokens = 512
	}
	if maxTokens > ideal {
		return ideal
	}

	// Round down to nearest 128
	maxTokens = (maxTokens / 128) * 128
	if maxTokens < 128 {
		maxTokens = 128
	}
	return maxTokens
}

func reservedForBytes(bytesPerToken uint64, availableBytes uint64) int {
	if bytesPerToken == 0 {
		return 0
	}
	return int(availableBytes / bytesPerToken)
}

func buildCommand(llamaServerPath string, flags []string) string {
	parts := []string{llamaServerPath}
	for _, f := range flags {
		parts = append(parts, f)
	}
	return strings.Join(parts, " ")
}

// FormatMB returns a human-readable string for MiB.
func FormatMB(mb uint64) string {
	if mb >= 1024 {
		return fmt.Sprintf("%.1f GB", float64(mb)/1024.0)
	}
	return fmt.Sprintf("%d MB", mb)
}

// GPUReasoning returns a human-readable explanation of GPU selection.
func GPUReasoning(inv *hardware.Inventory, requiredVRAM uint64, backend ...runtimemgr.BuildBackend) string {
	var lines []string
	effectiveBackend := normalizeBackend(backend...)
	if inv == nil {
		inv = &hardware.Inventory{}
	}
	switch effectiveBackend {
	case runtimemgr.BuildBackendCPU:
		return "Runtime backend is CPU-only — will use CPU"
	case runtimemgr.BuildBackendVulkan:
		candidates := usableVulkanGPUs(inv.VulkanGPUs)
		if len(candidates) == 0 {
			return "No Vulkan GPU detected — will use CPU"
		}
		lines = append(lines, fmt.Sprintf("Required VRAM: %s (model %s + 20%% KV cache overhead)",
			FormatMB(requiredVRAM), FormatMB(requiredVRAM-(requiredVRAM*20/100))))
		bestGPU, _, availableVRAM := selectBestVulkanGPU(inv.VulkanGPUs, requiredVRAM)
		if bestGPU != nil {
			lines = append(lines, fmt.Sprintf("Selected: GPU %d (%s) with %s free (%s headroom)",
				bestGPU.Index,
				bestGPU.Name,
				FormatMB(availableVRAM),
				FormatMB(availableVRAM-requiredVRAM)))
		} else {
			for _, g := range candidates {
				lines = append(lines, fmt.Sprintf("  GPU %d (%s): %s free, insufficient (need %s)",
					g.Index,
					g.Name,
					FormatMB(g.FreeVRAM),
					FormatMB(requiredVRAM)))
			}
			lines = append(lines, "No Vulkan GPU with enough VRAM — will use CPU fallback")
		}
		return strings.Join(lines, "\n")
	}
	gpus := inv.GPUs
	gpuLabel := "GPU"
	if effectiveBackend == runtimemgr.BuildBackendROCm {
		gpus = inv.ROCmGPUs
		gpuLabel = "ROCm GPU"
	}
	if len(gpus) == 0 {
		return fmt.Sprintf("No %s detected — will use CPU", gpuLabel)
	}
	lines = append(lines, fmt.Sprintf("Required VRAM: %s (model %s + 20%% KV cache overhead)",
		FormatMB(requiredVRAM), FormatMB(requiredVRAM-(requiredVRAM*20/100))))

	bestGPU, _, availableVRAM := selectBestGPU(gpus, requiredVRAM)
	if bestGPU != nil {
		lines = append(lines, fmt.Sprintf("Selected: GPU %d (%s) with %s free (%s headroom)",
			bestGPU.Index,
			bestGPU.DisplayName(),
			FormatMB(availableVRAM),
			FormatMB(availableVRAM-requiredVRAM)))
	} else {
		for _, g := range gpus {
			lines = append(lines, fmt.Sprintf("  GPU %d (%s): %s free, insufficient (need %s)",
				g.Index,
				g.DisplayName(),
				FormatMB(g.VRAMFree),
				FormatMB(requiredVRAM)))
		}
		lines = append(lines, "No GPU with enough VRAM — will use CPU fallback")
	}
	return strings.Join(lines, "\n")
}
