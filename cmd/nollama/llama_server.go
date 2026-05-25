package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/sovereignty-labs/nollama/internal/config"
	runtimemgr "github.com/sovereignty-labs/nollama/internal/runtime"
	"github.com/spf13/cobra"
)

func resolveLlamaServerPath(cmd *cobra.Command, cfg *config.Config) (string, error) {
	return resolveLlamaServerPathWithRuntime(cmd, cfg, getRuntimeFlag(cmd))
}

func resolveLlamaServerPathWithRuntime(cmd *cobra.Command, cfg *config.Config, runtimeName string) (string, error) {
	if path := getLlamaServerFlag(cmd); path != "" {
		return path, nil
	}

	if runtimeName = strings.TrimSpace(runtimeName); runtimeName != "" {
		path, err := runtimemgr.NewManager().ResolveNamed(runtimeName)
		if err != nil {
			return "", err
		}
		return path, nil
	}

	if path := os.Getenv("NOLLAMA_LLAMA_SERVER"); path != "" {
		return path, nil
	}

	if cfg != nil && cfg.LlamaServer != "" {
		return cfg.LlamaServer, nil
	}

	if cfg == nil {
		if cfgPath := config.FindConfig(); cfgPath != "" {
			loaded, err := config.Load(cfgPath)
			if err != nil {
				return "", err
			}
			if loaded.LlamaServer != "" {
				return loaded.LlamaServer, nil
			}
		}
	}

	path, err := runtimemgr.NewManager().Resolve()
	if err == nil {
		return path, nil
	}

	return "", fmt.Errorf("no llama-server found. Run `nollama runtime install` or set --llama-server")
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
