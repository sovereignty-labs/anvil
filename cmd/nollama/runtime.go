package main

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	runtimemgr "github.com/sovereignty-labs/nollama/internal/runtime"
	"github.com/spf13/cobra"
)

var runtimeInstallVersion string
var runtimeInstallNoPrebuilt bool
var runtimeInstallNoBuild bool
var (
	runtimeBuildRepo    string
	runtimeBuildBranch  string
	runtimeBuildName    string
	runtimeBuildBackend string
	runtimeAddBackend   string
)

func init() {
	runtimeInstallCmd.Flags().StringVar(&runtimeInstallVersion, "version", "", "Install a specific llama.cpp release tag")
	runtimeInstallCmd.Flags().BoolVar(&runtimeInstallNoPrebuilt, "no-prebuilt", false, "Skip sovereignty-labs pre-built runtimes and fall back to ggml.org or source builds")
	runtimeInstallCmd.Flags().BoolVar(&runtimeInstallNoBuild, "no-build", false, "Download only; do not auto-build from source when no GPU binary is available")

	runtimeBuildCmd.Flags().StringVar(&runtimeBuildRepo, "repo", "", "Git repository to clone (default: ggml-org/llama.cpp)")
	runtimeBuildCmd.Flags().StringVar(&runtimeBuildBranch, "branch", "", "Branch or tag to checkout")
	runtimeBuildCmd.Flags().StringVar(&runtimeBuildName, "name", "", "Runtime name (default: llama-build, llama-vulkan, or llama-cpu)")
	runtimeBuildCmd.Flags().StringVar(&runtimeBuildBackend, "backend", "", "GPU backend to use (cuda, vulkan, cpu; default: auto-detect)")
	runtimeBuildCmd.RunE = runRuntimeBuild
	runtimeBuildCmd.Short = "Build llama-server from source"
	runtimeBuildCmd.Long = "Clone and compile llama.cpp (or a fork) from source. Requires git, cmake, and a C/C++ compiler. Backend auto-detection prefers CUDA (nvcc), then Vulkan, then CPU-only."
	runtimeAddCmd.Flags().StringVar(&runtimeAddBackend, "backend", "", "Backend override for the added runtime (cuda, rocm, vulkan, cpu; default: auto-detect from shared libraries)")
}

func runRuntimeInstall(_ *cobra.Command, _ []string) error {
	mgr := runtimemgr.NewManager()
	_, err := mgr.Install(runtimeInstallVersion, runtimeInstallNoPrebuilt, runtimeInstallNoBuild)
	return err
}

func runRuntimeBuild(_ *cobra.Command, _ []string) error {
	mgr := runtimemgr.NewManager()
	_, err := mgr.Build(runtimemgr.BuildOpts{
		Repo:    runtimeBuildRepo,
		Branch:  runtimeBuildBranch,
		Name:    runtimeBuildName,
		Backend: runtimeBuildBackend,
	})
	return err
}

func runRuntimeList(_ *cobra.Command, _ []string) error {
	mgr := runtimemgr.NewManager()
	runtimes, err := mgr.List()
	if err != nil {
		return err
	}

	if len(runtimes) == 0 {
		fmt.Println("No runtimes installed. Run `nollama runtime install` to add one.")
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "NAME\tVERSION\tSOURCE\tACTIVE")
	for _, rt := range runtimes {
		version := "—"
		if rt.Version != "" {
			version = rt.Version
		}
		source := rt.Source
		if source == "" {
			source = "custom"
		}
		active := ""
		if rt.Active {
			active = "✓"
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", rt.Name, version, source, active)
	}
	return w.Flush()
}

func runRuntimeUse(_ *cobra.Command, args []string) error {
	mgr := runtimemgr.NewManager()
	runtimes, err := mgr.List()
	if err != nil {
		return err
	}

	oldActive := "none"
	for _, rt := range runtimes {
		if rt.Active {
			oldActive = rt.Name
			break
		}
	}

	if err := mgr.Use(args[0]); err != nil {
		return err
	}

	fmt.Printf("Active runtime: %s -> %s\n", oldActive, args[0])
	return nil
}

func runRuntimeAdd(_ *cobra.Command, args []string) error {
	mgr := runtimemgr.NewManager()
	if err := mgr.Add(args[0], args[1], runtimemgr.BuildBackend(strings.ToLower(strings.TrimSpace(runtimeAddBackend)))); err != nil {
		return err
	}

	runtimes, err := mgr.List()
	if err != nil {
		return err
	}
	for _, rt := range runtimes {
		if rt.Name == args[0] {
			fmt.Printf("Added runtime %q (copied to %s)\n", args[0], rt.Path)
			return nil
		}
	}

	fmt.Printf("Added runtime %q\n", args[0])
	return nil
}
