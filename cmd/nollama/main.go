package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/hirdforge/nollama/internal/hardware"
	"github.com/hirdforge/nollama/internal/model"
	"github.com/hirdforge/nollama/internal/process"
	"github.com/hirdforge/nollama/internal/version"
	"github.com/spf13/cobra"
)

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

var rootCmd = &cobra.Command{
	Use:   "nollama",
	Short: "The model runner Ollama should have been",
	Long: `nollama — One Go binary. Plain GGUFs. Transparent llama-server under the hood.

nollama manages llama-server processes with smart defaults derived from GGUF
metadata and hardware detection. Zero inference overhead. Plain files.
No blob store. No proprietary formats. No cloud.

Powered by llama.cpp (Georgi Gerganov). MIT licensed.`,
	SilenceUsage:  true,
	SilenceErrors: true,
	Version:       version.Version,
}

// --- serve ---

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the nollama daemon",
	Long:  "Run nollama as a daemon, managing llama-server processes and serving the unified API.",
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2")
	},
}

// --- load ---

var loadCmd = &cobra.Command{
	Use:   "load <model.gguf>",
	Short: "Load a model (spawn a llama-server process)",
	Long: `Load a model and start serving it via llama-server.

With --dry-run, prints the computed flags and device selection reasoning
without actually launching llama-server.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runLoad(cmd, args)
	},
}

// --- unload ---

var unloadCmd = &cobra.Command{
	Use:   "unload <model.gguf>",
	Short: "Unload a model (kill the llama-server process)",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2")
	},
}

// --- status ---

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show fleet status — nodes, GPUs, loaded models, VRAM",
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2")
	},
}

// --- models ---

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "List local GGUF models",
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2")
	},
}

// --- inspect ---

var inspectCmd = &cobra.Command{
	Use:   "inspect <model.gguf>",
	Short: "Show GGUF metadata and hardware recommendations",
	Args:  cobra.ExactArgs(1),
	RunE:  runInspect,
}

// --- pull ---

var pullCmd = &cobra.Command{
	Use:   "pull <org/repo:quant>",
	Short: "Pull a GGUF from HuggingFace",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2+")
	},
}

// --- runtime ---

var runtimeCmd = &cobra.Command{
	Use:   "runtime",
	Short: "Manage llama-server runtimes (install, build, switch)",
}

var runtimeInstallCmd = &cobra.Command{
	Use:   "install",
	Short: "Download pre-built llama-server from GitHub Releases",
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2+")
	},
}

var runtimeBuildCmd = &cobra.Command{
	Use:   "build",
	Short: "Compile llama-server from source (including forks)",
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2+")
	},
}

var runtimeListCmd = &cobra.Command{
	Use:   "list",
	Short: "List installed llama-server runtimes",
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2+")
	},
}

var runtimeUseCmd = &cobra.Command{
	Use:   "use <name>",
	Short: "Set the active llama-server runtime",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2+")
	},
}

var runtimeAddCmd = &cobra.Command{
	Use:   "add <name> <path>",
	Short: "Register an existing llama-server binary",
	Args:  cobra.ExactArgs(2),
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2+")
	},
}

// --- remote ---

var remoteCmd = &cobra.Command{
	Use:   "remote",
	Short: "Manage remote nollama nodes (federation)",
}

var remoteAddCmd = &cobra.Command{
	Use:   "add <name> <url>",
	Short: "Register a remote nollama node",
	Args:  cobra.ExactArgs(2),
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — Phase 2")
	},
}

var remoteListCmd = &cobra.Command{
	Use:   "list",
	Short: "List registered remote nodes",
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — Phase 2")
	},
}

var remoteRmCmd = &cobra.Command{
	Use:   "rm <name>",
	Short: "Remove a remote node",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — Phase 2")
	},
}

var remotePingCmd = &cobra.Command{
	Use:   "ping",
	Short: "Health-check all remote nodes",
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — Phase 2")
	},
}

// --- rm ---

var rmCmd = &cobra.Command{
	Use:   "rm <model.gguf>",
	Short: "Remove a downloaded model",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2+")
	},
}

// --- cp ---

var cpCmd = &cobra.Command{
	Use:   "cp <model.gguf> --to <node>",
	Short: "Copy a model to another node",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — Phase 2")
	},
}

// runLoad handles the load command — parses GGUF, detects hardware, computes flags.
func runLoad(cmd *cobra.Command, args []string) error {
	modelPath := args[0]

	// Resolve llama-server path: try local flag, then persistent flag, then env var
	llamaServerFlag, _ := cmd.Flags().GetString("llama-server")
	if llamaServerFlag == "" {
		llamaServerFlag, _ = cmd.Parent().PersistentFlags().GetString("llama-server")
	}
	if llamaServerFlag == "" {
		llamaServerFlag = os.Getenv("NOLLAMA_LLAMA_SERVER")
	}
	if llamaServerFlag == "" {
		return fmt.Errorf("llama-server path is required: set --llama-server flag or NOLLAMA_LLAMA_SERVER env var")
	}

	// Parse GGUF metadata
	fmt.Println("Parsing GGUF metadata...")
	meta, err := model.ParseGGUF(modelPath)
	if err != nil {
		return fmt.Errorf("failed to parse GGUF: %w", err)
	}

	fmt.Printf("  Model: %s (%s %s)\n", meta.Name, meta.ArchDisplayName(), meta.QuantName)
	fmt.Printf("  Size:  %.1f GB, Context: %d tokens\n", meta.FileSizeGB(), meta.ContextLength)

	// Detect hardware
	fmt.Println("Detecting hardware...")
	inv, err := hardware.Detect()
	if err != nil {
		return fmt.Errorf("hardware detection failed: %w", err)
	}
	fmt.Printf("  GPUs:  %d detected\n", len(inv.GPUs))
	if len(inv.GPUs) > 0 {
		for _, g := range inv.GPUs {
			fmt.Printf("    GPU %d: %s — %.1f GB total, %.1f GB free\n",
				g.Index, g.DisplayName(), g.VRAMTotalGB(), g.VRAMFreeGB())
		}
	}
	fmt.Printf("  CPU:   %s — %d cores, %d threads\n",
		inv.CPU.ModelName, inv.CPU.Cores, inv.CPU.Threads)

	// Compute flags
	fmt.Println("Computing flags...")
	dryRun, _ := cmd.Flags().GetBool("dry-run")

	result, err := process.ComputeFlags(meta, modelPath, inv, llamaServerFlag, 0)
	if err != nil {
		return fmt.Errorf("flag computation failed: %w", err)
	}

	if dryRun {
		fmt.Println()
		fmt.Println("=== Dry Run: Computed llama-server command ===")
		fmt.Println()

		// Device selection reasoning
		fileSizeMB := uint64(meta.FileSizeBytes) / 1024 / 1024
		requiredVRAM := fileSizeMB + (fileSizeMB * 20) / 100
		fmt.Println(GPUReasoning(inv, requiredVRAM))
		fmt.Println()

		// Device summary
		if result.CPUFallback {
			fmt.Printf("Device:    %s (CPU fallback — no GPU has enough VRAM)\n", result.SelectedDevice)
			fmt.Printf("Threads:   %d\n", result.CPUThreads)
		} else {
			fmt.Printf("Device:    %s\n", result.SelectedDevice)
			fmt.Printf("VRAM:      %s used / %s available\n", process.FormatMB(result.VRAMUsedMB), process.FormatMB(result.VRAMTotalMB))
			fmt.Printf("GPU Layers: all (99)\n")
		}

		// Context size note
		if meta.ContextLength > 0 {
			foundCtx := false
			for i, f := range result.Flags {
				if f == "--ctx-size" && i+1 < len(result.Flags) {
					fmt.Printf("Context:   %s (capped from %d by VRAM)\n", result.Flags[i+1], meta.ContextLength)
					foundCtx = true
					break
				}
			}
			if !foundCtx {
				fmt.Printf("Context:   %d (from GGUF metadata)\n", meta.ContextLength)
			}
		}

		// Flags table
		fmt.Println()
		fmt.Println("Flags:")
		for i := 0; i < len(result.Flags); i++ {
			flag := result.Flags[i]
			if i+1 < len(result.Flags) && !strings.HasPrefix(result.Flags[i+1], "--") {
				fmt.Printf("  %-20s %s\n", flag+":", result.Flags[i+1])
				i++
			} else {
				fmt.Printf("  %-20s (true)\n", flag+":")
			}
		}

		// Full command
		fmt.Println()
		fmt.Printf("Command:\n  %s\n", result.Command)
		fmt.Println()
		return nil
	}

	fmt.Println()
	fmt.Printf("Device:    %s\n", result.SelectedDevice)
	fmt.Printf("Command:   %s\n", result.Command)
	fmt.Println()
	fmt.Println("not implemented yet — coming next session")
	fmt.Println("Run with --dry-run to see computed flags for this model.")
	return nil
}

// GPUReasoning is a wrapper for process.GPUReasoning.
func GPUReasoning(inv *hardware.Inventory, requiredVRAM uint64) string {
	return process.GPUReasoning(inv, requiredVRAM)
}

// FormatMB is a wrapper for process.FormatMB.
func FormatMB(mb uint64) string {
	return process.FormatMB(mb)
}

func init() {
	// Root-level persistent flags
	rootCmd.PersistentFlags().String("node", "", "Target a specific remote node")
	rootCmd.PersistentFlags().String("llama-server", "", "Path to llama-server binary (required; also checked via NOLLAMA_LLAMA_SERVER env var)")

	// Register subcommands
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(loadCmd)
	rootCmd.AddCommand(unloadCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(modelsCmd)
	rootCmd.AddCommand(inspectCmd)
	rootCmd.AddCommand(pullCmd)
	rootCmd.AddCommand(runtimeCmd)
	rootCmd.AddCommand(remoteCmd)
	rootCmd.AddCommand(rmCmd)
	rootCmd.AddCommand(cpCmd)

	// runtime subcommands
	runtimeCmd.AddCommand(runtimeInstallCmd)
	runtimeCmd.AddCommand(runtimeBuildCmd)
	runtimeCmd.AddCommand(runtimeListCmd)
	runtimeCmd.AddCommand(runtimeUseCmd)
	runtimeCmd.AddCommand(runtimeAddCmd)

	// remote subcommands
	remoteCmd.AddCommand(remoteAddCmd)
	remoteCmd.AddCommand(remoteListCmd)
	remoteCmd.AddCommand(remoteRmCmd)
	remoteCmd.AddCommand(remotePingCmd)

	// load flags
	loadCmd.Flags().Int("gpu", -1, "GPU index to load on")
	loadCmd.Flags().Bool("cpu", false, "Force CPU inference")
	loadCmd.Flags().String("runtime", "", "Use a specific llama-server runtime")
	loadCmd.Flags().String("profile", "", "Apply a hardware profile")
	loadCmd.Flags().Bool("dry-run", false, "Show what would be passed to llama-server")
	loadCmd.Flags().Bool("swap", false, "Evict LRU model if VRAM is full")

	// serve flags
	serveCmd.Flags().String("bind", "0.0.0.0:11434", "Listen address")
	serveCmd.Flags().String("config", "", "Config file path")
	serveCmd.Flags().Bool("mcp", false, "Enable MCP server")

	// cp flags
	cpCmd.Flags().String("to", "", "Target node for copy")
}