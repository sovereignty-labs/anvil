package main

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/sovereignty-labs/nollama/internal/hardware"
	"github.com/sovereignty-labs/nollama/internal/model"
	"github.com/spf13/cobra"
)

func runInspect(cmd *cobra.Command, args []string) error {
	path, err := resolveModelPath(args[0])
	if err != nil {
		return err
	}

	// Parse GGUF metadata
	meta, err := model.ParseGGUF(path)
	if err != nil {
		return fmt.Errorf("failed to parse GGUF: %w", err)
	}

	filename := filepath.Base(path)

	// Print model info
	fmt.Println()
	printField("Model", strings.TrimSuffix(filename, ".gguf"))
	printField("Arch", meta.ArchDisplayName())
	printField("Quant", meta.QuantDisplayName(filename))
	printField("Size", fmt.Sprintf("%.1f GB", meta.FileSizeGB()))

	if meta.ContextLength > 0 {
		printField("Context", fmt.Sprintf("%s (embedded)", formatNumber(meta.ContextLength)))
	}

	if meta.Name != "" {
		printField("Name", meta.Name)
	}

	if meta.HasChatTemplate {
		printField("Template", "Jinja ✓ (embedded in GGUF)")
	} else {
		printField("Template", "None (llama-server default)")
	}

	if meta.EmbeddingLength > 0 {
		printField("Embedding", fmt.Sprintf("%d", meta.EmbeddingLength))
	}
	if meta.BlockCount > 0 {
		printField("Blocks", fmt.Sprintf("%d", meta.BlockCount))
	}
	if meta.HeadCount > 0 {
		heads := fmt.Sprintf("%d", meta.HeadCount)
		if meta.HeadCountKV > 0 && meta.HeadCountKV != meta.HeadCount {
			heads += fmt.Sprintf(" (KV: %d)", meta.HeadCountKV)
		}
		printField("Heads", heads)
	}

	if meta.ParamCount > 0 {
		printField("Params", formatParamCount(meta.ParamCount))
	}

	printField("GGUF Ver", fmt.Sprintf("%d", meta.Version))
	printField("Tensors", formatNumber(meta.TensorCount))

	// Hardware detection
	fmt.Println()
	inv, err := hardware.Detect()
	if err != nil {
		fmt.Printf("  ⚠ Hardware detection failed: %v\n", err)
	} else {
		fmt.Println("  Available hardware:")
		if len(inv.GPUs) > 0 {
			for _, gpu := range inv.GPUs {
				fmt.Printf("    GPU %d: %-30s — %s MiB (%s free)\n",
					gpu.Index,
					gpu.DisplayName(),
					formatNumber(gpu.VRAMTotal),
					formatNumber(gpu.VRAMFree),
				)
			}
		} else {
			fmt.Println("    No NVIDIA GPUs detected")
		}
		fmt.Printf("    CPU:   %.0f GB RAM (%.0f GB free), %d threads\n",
			inv.CPU.RAMTotalGB(),
			inv.CPU.RAMFreeGB(),
			inv.CPU.Threads,
		)

		// Simple recommendation
		if len(inv.GPUs) > 0 {
			fileSizeMB := uint64(meta.FileSizeBytes / (1024 * 1024))
			// VRAM estimate: file size + ~20% overhead for KV cache at default context
			estimatedMB := fileSizeMB + (fileSizeMB / 5)

			fmt.Println()
			bestGPU := -1
			bestFree := uint64(0)
			for _, gpu := range inv.GPUs {
				if gpu.VRAMFree >= estimatedMB && gpu.VRAMFree > bestFree {
					bestGPU = gpu.Index
					bestFree = gpu.VRAMFree
				}
			}
			if bestGPU >= 0 {
				gpu := inv.GPUs[bestGPU]
				headroom := float64(gpu.VRAMFree-estimatedMB) / 1024.0
				fmt.Printf("  Recommendation: GPU %d (%s, %.1f GB estimated, %.1f GB headroom)\n",
					bestGPU, gpu.DisplayName(),
					float64(estimatedMB)/1024.0, headroom,
				)
			} else {
				// Check if it fits in CPU RAM
				if inv.CPU.RAMFreeMB >= estimatedMB {
					fmt.Printf("  Recommendation: CPU (no GPU has enough VRAM; %.1f GB estimated, CPU has %.0f GB free)\n",
						float64(estimatedMB)/1024.0, inv.CPU.RAMFreeGB(),
					)
				} else {
					fmt.Printf("  ⚠ No device has enough memory (%.1f GB estimated)\n",
						float64(estimatedMB)/1024.0,
					)
				}
			}
		}
	}

	fmt.Println()
	return nil
}

func printField(label, value string) {
	fmt.Printf("  %-12s %s\n", label+":", value)
}

func formatNumber(n uint64) string {
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		return s
	}
	// Insert commas
	var result []byte
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			result = append(result, ',')
		}
		result = append(result, byte(c))
	}
	return string(result)
}

func formatParamCount(n uint64) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(n)/1_000_000_000)
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}
