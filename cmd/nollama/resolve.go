package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sovereignty-labs/nollama/internal/config"
)

// resolveModelPath resolves a bare filename (no path separator) against the
// known model directories so users can refer to models pulled into the default
// store by name alone:
//
//	nollama load google_gemma-4-E2B-it-Q4_K_M.gguf
//	nollama inspect google_gemma-4-E2B-it-Q4_K_M       # .gguf auto-appended
//
// Search order: current working directory → config.ModelDir → default
// (~/.local/share/nollama/models). The first existing file wins. Paths that
// contain a separator or are absolute are returned untouched.
func resolveModelPath(path string) (string, error) {
	if filepath.IsAbs(path) || strings.ContainsRune(path, os.PathSeparator) {
		return path, nil
	}

	candidates := []string{path}
	if !strings.HasSuffix(strings.ToLower(path), ".gguf") {
		candidates = append(candidates, path+".gguf")
	}

	var searched []string

	// 1. Current working directory.
	for _, name := range candidates {
		if _, err := os.Stat(name); err == nil {
			abs, _ := filepath.Abs(name)
			return abs, nil
		}
	}
	if cwd, err := os.Getwd(); err == nil {
		searched = append(searched, cwd)
	}

	// 2. Config-defined model_dir.
	if cfgPath := config.FindConfig(); cfgPath != "" {
		if cfg, err := config.Load(cfgPath); err == nil && cfg.ModelDir != "" {
			for _, name := range candidates {
				candidate := filepath.Join(cfg.ModelDir, name)
				if _, err := os.Stat(candidate); err == nil {
					return candidate, nil
				}
			}
			searched = append(searched, cfg.ModelDir)
		}
	}

	// 3. Default model directory.
	defaultDir := config.DefaultConfig().ModelDir
	for _, name := range candidates {
		candidate := filepath.Join(defaultDir, name)
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}
	if defaultDir != "" {
		searched = append(searched, defaultDir)
	}

	return "", fmt.Errorf("model not found: %s\nSearched: %s\nPull a model with: nollama pull <org>/<repo>:<quant>",
		path, strings.Join(searched, ", "))
}
