package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/sovereignty-labs/anvil/internal/config"
	runtimemgr "github.com/sovereignty-labs/anvil/internal/runtime"
	"github.com/spf13/cobra"
)

func resolveLlamaServerPath(cmd *cobra.Command, cfg *config.Config) (string, error) {
	path, _, err := resolveLlamaServerPathWithRuntime(cmd, cfg, getRuntimeFlag(cmd))
	return path, err
}

func resolveLlamaServerPathWithRuntime(cmd *cobra.Command, cfg *config.Config, runtimeName string) (string, runtimemgr.BuildBackend, error) {
	if path := getLlamaServerFlag(cmd); path != "" {
		return path, runtimemgr.BuildBackendCUDA, nil
	}

	if runtimeName = strings.TrimSpace(runtimeName); runtimeName != "" {
		mgr := runtimemgr.NewManager()
		runtimesDir, dirErr := mgr.RuntimesDir()
		if dirErr != nil {
			return "", runtimemgr.BuildBackendCUDA, dirErr
		}
		path, err := mgr.ResolveNamed(runtimeName)
		if err != nil {
			return "", runtimemgr.BuildBackendCUDA, fmt.Errorf("runtime %q not found in %s: %w", runtimeName, runtimesDir, err)
		}
		return path, mgr.RuntimeBackend(runtimeName), nil
	}

	if path := os.Getenv("ANVIL_LLAMA_SERVER"); path != "" {
		return path, runtimemgr.BuildBackendCUDA, nil
	}

	if cfg != nil && cfg.LlamaServer != "" {
		return cfg.LlamaServer, runtimemgr.BuildBackendCUDA, nil
	}

	if cfg == nil {
		if cfgPath := config.FindConfig(); cfgPath != "" {
			loaded, err := config.Load(cfgPath)
			if err != nil {
				return "", runtimemgr.BuildBackendCUDA, err
			}
			if loaded.LlamaServer != "" {
				return loaded.LlamaServer, runtimemgr.BuildBackendCUDA, nil
			}
		}
	}

	mgr := runtimemgr.NewManager()
	runtimesDir, dirErr := mgr.RuntimesDir()
	if dirErr != nil {
		return "", runtimemgr.BuildBackendCUDA, dirErr
	}
	activeName, err := mgr.ActiveName()
	if err != nil {
		return "", runtimemgr.BuildBackendCUDA, err
	}
	path, err := mgr.Resolve()
	if err == nil {
		return path, mgr.RuntimeBackend(activeName), nil
	}

	return "", runtimemgr.BuildBackendCUDA, fmt.Errorf("no llama-server found in %s. Run `anvil runtime install` or set --llama-server", runtimesDir)
}

func getRuntimeFlag(cmd *cobra.Command) string {
	if cmd == nil {
		return ""
	}
	path, _ := cmd.Flags().GetString("runtime")
	return path
}

func getLlamaServerFlag(cmd *cobra.Command) string {
	if cmd == nil {
		return ""
	}
	if path, _ := cmd.Flags().GetString("llama-server"); path != "" {
		return path
	}
	if cmd.Parent() != nil {
		if path, _ := cmd.Parent().PersistentFlags().GetString("llama-server"); path != "" {
			return path
		}
	}
	return ""
}
