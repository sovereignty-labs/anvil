package main

import (
	"fmt"
	"os"
	"text/tabwriter"

	runtimemgr "github.com/hirdforge/nollama/internal/runtime"
	"github.com/spf13/cobra"
)

var runtimeInstallVersion string

func init() {
	runtimeInstallCmd.Flags().StringVar(&runtimeInstallVersion, "version", "", "Install a specific llama.cpp release tag")
}

func runRuntimeInstall(_ *cobra.Command, _ []string) error {
	mgr := runtimemgr.NewManager()
	_, err := mgr.Install(runtimeInstallVersion)
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
	if err := mgr.Add(args[0], args[1]); err != nil {
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
