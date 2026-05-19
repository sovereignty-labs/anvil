// Package process computes llama-server flags from GGUF metadata and hardware inventory.
package process

import (
	"fmt"
	"strings"

	"github.com/hirdforge/nollama/internal/hardware"
	"github.com/hirdforge/nollama/internal/model"
)

// Result holds the computed flags, device selection, and metadata for a model load.
type Result struct {
	SelectedDevice string   // "cuda:0", "cuda:1", "cpu", etc.
	Flags          []string // llama-server flags as []string
	Command        string   // full command line string
	VRAMUsedMB     uint64   // estimated VRAM usage in MiB
	VRAMTotalMB    uint64   // available VRAM on selected device in MiB
	CPUFallback    bool     // whether CPU fallback was used
	CPUThreads     int      // number of CPU threads (non-zero only when CPU fallback)
	GPUIndex       int      // GPU index used (-1 for CPU fallback)
	Port           int      // llama-server HTTP port (11434 + modelIndex)
}

// EnsureHostFlag injects --host 0.0.0.0 into flags when no --host is already
// present. llama-server itself defaults to 127.0.0.1, which makes spawned
// instances unreachable from federation peers or other machines on the LAN —
// the inverse of what nollama is for. Called on the final merged flag slice
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
func ComputeFlags(meta *model.GGUFMetadata, modelPath string, inv *hardware.Inventory, llamaServerPath string, modelIndex int) (*Result, error) {
	if llamaServerPath == "" {
		return nil, fmt.Errorf("llama-server path is required")
	}
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}

	port := 11434 + modelIndex

	result := &Result{
		Flags: []string{
			"--model", modelPath,
			"--host", "0.0.0.0",
			"--port", fmt.Sprintf("%d", port),
		},
		Port: port,
	}

	// Estimate required VRAM: file size + 20% overhead for KV cache
	fileSizeMB := uint64(meta.FileSizeBytes) / 1024 / 1024
	requiredVRAM := fileSizeMB + (fileSizeMB * 20) / 100
	result.VRAMUsedMB = requiredVRAM

	// Try to find the best GPU
	bestGPU, _, availableVRAM := selectBestGPU(inv.GPUs, requiredVRAM)

	if bestGPU != nil {
		result.SelectedDevice = fmt.Sprintf("cuda:%d", bestGPU.Index)
		result.GPUIndex = bestGPU.Index
		result.Flags = append(result.Flags,
			"--n-gpu-layers", "99",
			"--flash-attn", "on",
			"--no-warmup",
		)
		result.VRAMTotalMB = availableVRAM

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
	} else {
		// CPU fallback
		result.CPUFallback = true
		result.SelectedDevice = "cpu"
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

	// Build the full command line
	result.Command = buildCommand(llamaServerPath, result.Flags)

	return result, nil
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
func GPUReasoning(inv *hardware.Inventory, requiredVRAM uint64) string {
	var lines []string
	if len(inv.GPUs) == 0 {
		return "No GPU detected — will use CPU"
	}
	lines = append(lines, fmt.Sprintf("Required VRAM: %s (model %s + 20%% KV cache overhead)",
		FormatMB(requiredVRAM), FormatMB(requiredVRAM-(requiredVRAM*20/100))))

	bestGPU, _, availableVRAM := selectBestGPU(inv.GPUs, requiredVRAM)
	if bestGPU != nil {
		lines = append(lines, fmt.Sprintf("Selected: GPU %d (%s) with %s free (%s headroom)",
			bestGPU.Index,
			bestGPU.DisplayName(),
			FormatMB(availableVRAM),
			FormatMB(availableVRAM-requiredVRAM)))
	} else {
		for _, g := range inv.GPUs {
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
