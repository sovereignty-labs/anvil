package main

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/hirdforge/nollama/internal/config"
	"github.com/hirdforge/nollama/internal/model"
	"github.com/spf13/cobra"
)

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "List available GGUF models",
	Long: `List all GGUF files in the model directory with metadata.

Shows filename, size, architecture, quantization, and context length
read from GGUF headers.

The model directory is determined by:
  1. --model-dir flag
  2. NOLLAMA_MODEL_DIR environment variable
  3. config file model_dir setting
  4. Default: ~/.local/share/nollama/models`,
	RunE: runModels,
}

var modelsDir string

func init() {
	modelsCmd.Flags().StringVar(&modelsDir, "model-dir", "", "Override model directory")
}

func runModels(cmd *cobra.Command, args []string) error {
	// Determine model directory
	dir := modelsDir
	if dir == "" {
		dir = os.Getenv("NOLLAMA_MODEL_DIR")
	}
	if dir == "" {
		// Try config file
		cfgPath := config.FindConfig()
		if cfgPath != "" {
			if cfg, err := config.Load(cfgPath); err == nil && cfg.ModelDir != "" {
				dir = cfg.ModelDir
			}
		}
	}
	if dir == "" {
		dir = config.DefaultConfig().ModelDir
	}

	// Check directory exists
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return fmt.Errorf("model directory does not exist: %s\nCreate it or set model_dir in config", dir)
	}

	// Scan for models
	models, err := model.ScanDir(dir)
	if err != nil {
		return fmt.Errorf("scanning models: %w", err)
	}

	if len(models) == 0 {
		fmt.Printf("No GGUF models found in %s\n", dir)
		fmt.Println("Pull models with: nollama pull <org>/<repo>:<quant>")
		return nil
	}

	// Print table
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintf(w, "MODEL\tSIZE\tARCH\tQUANT\tCONTEXT\n")

	for _, m := range models {
		arch := "-"
		quant := "-"
		ctx := "-"

		if m.Meta != nil {
			if m.Meta.Architecture != "" {
				arch = m.Meta.Architecture
			}
			if m.Meta.QuantName != "" {
				quant = m.Meta.QuantName
			}
			if m.Meta.ContextLength > 0 {
				ctx = formatContext(m.Meta.ContextLength)
			}
		}

		// Strip .gguf for cleaner display
		name := strings.TrimSuffix(m.Filename, ".gguf")

		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", name, m.SizeHuman(), arch, quant, ctx)
	}

	w.Flush()
	fmt.Printf("\n%d models, %s (%s)\n", len(models), totalSize(models), dir)

	return nil
}

func formatContext(ctx uint64) string {
	if ctx >= 1024*1024 {
		return fmt.Sprintf("%.0fM", float64(ctx)/(1024*1024))
	}
	if ctx >= 1024 {
		return fmt.Sprintf("%dK", ctx/1024)
	}
	return fmt.Sprintf("%d", ctx)
}

func totalSize(models []model.ModelInfo) string {
	var total int64
	for _, m := range models {
		total += m.SizeBytes
	}
	const gb = 1024 * 1024 * 1024
	if total >= gb {
		return fmt.Sprintf("%.1f GB", float64(total)/float64(gb))
	}
	const mb = 1024 * 1024
	return fmt.Sprintf("%.0f MB", float64(total)/float64(mb))
}
