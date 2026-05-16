package main

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/hirdforge/nollama/internal/config"
	"github.com/hirdforge/nollama/internal/federation"
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
	if client, err := resolveNodeClient(cmd); err != nil {
		return err
	} else if client != nil {
		resp, err := client.Models()
		if err != nil {
			return err
		}
		if len(resp.Models) == 0 {
			nodeName, _ := cmd.Flags().GetString("node")
			fmt.Printf("No GGUF models found on %s\n", nodeName)
			return nil
		}
		renderRemoteModels(resp.Models)
		return nil
	}

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

	renderLocalModels(models)
	fmt.Printf("\n%d models, %s (%s)\n", len(models), totalSize(models), dir)

	return nil
}

type modelDisplay struct {
	Name    string
	Size    string
	Arch    string
	Quant   string
	Context string
}

func renderLocalModels(models []model.ModelInfo) {
	rows := make([]modelDisplay, 0, len(models))
	for _, m := range models {
		rows = append(rows, modelDisplay{
			Name:    strings.TrimSuffix(m.Filename, ".gguf"),
			Size:    m.SizeHuman(),
			Arch:    "-",
			Quant:   "-",
			Context: "-",
		})
		if m.Meta != nil {
			if m.Meta.Architecture != "" {
				rows[len(rows)-1].Arch = m.Meta.Architecture
			}
			if m.Meta.QuantName != "" {
				rows[len(rows)-1].Quant = m.Meta.QuantName
			}
			if m.Meta.ContextLength > 0 {
				rows[len(rows)-1].Context = formatContext(m.Meta.ContextLength)
			}
		}
	}
	renderModelTable(rows)
}

func renderRemoteModels(models []federation.ModelInfo) {
	rows := make([]modelDisplay, 0, len(models))
	for _, m := range models {
		row := modelDisplay{
			Name:    strings.TrimSuffix(m.Name, ".gguf"),
			Size:    m.SizeHuman,
			Arch:    "-",
			Quant:   "-",
			Context: "-",
		}
		if m.Arch != "" {
			row.Arch = m.Arch
		}
		if m.Quant != "" {
			row.Quant = m.Quant
		}
		if m.ContextLength > 0 {
			row.Context = formatContext(m.ContextLength)
		}
		rows = append(rows, row)
	}
	renderModelTable(rows)
}

func renderModelTable(rows []modelDisplay) {
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "MODEL\tSIZE\tARCH\tQUANT\tCONTEXT")
	for _, row := range rows {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", row.Name, row.Size, row.Arch, row.Quant, row.Context)
	}
	_ = w.Flush()
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
