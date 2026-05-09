package main

import (
	"fmt"
	"os"

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
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("not implemented yet — coming in S2")
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

func init() {
	// Root-level flags
	rootCmd.PersistentFlags().String("node", "", "Target a specific remote node")

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