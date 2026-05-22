package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/sovereignty-labs/nollama/internal/config"
	"github.com/sovereignty-labs/nollama/internal/server"
	"github.com/spf13/cobra"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Run nollama as a daemon",
	Long: `Run nollama as a long-running daemon that manages llama-server processes.

With --config, loads a config file and autoloads models on startup.
Without --config, starts with defaults and waits for load commands.

Signals:
  SIGTERM/SIGINT  Graceful shutdown (stops all models)
  SIGHUP          Reload config (reconciles model state)`,
	RunE: runServe,
}

var (
	serveConfigPath string
	serveBind       string
	serveMCP        bool
)

func init() {
	serveCmd.Flags().StringVar(&serveConfigPath, "config", "", "Path to config file")
	serveCmd.Flags().StringVar(&serveBind, "bind", "", "Listen address (overrides config)")
	serveCmd.Flags().BoolVar(&serveMCP, "mcp", false, "Enable the MCP server")
}

func runServe(cmd *cobra.Command, args []string) error {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	// Load config
	var cfg *config.Config
	var cfgPath string

	if serveConfigPath != "" {
		// Explicit config file
		var err error
		cfg, err = config.Load(serveConfigPath)
		if err != nil {
			return fmt.Errorf("load config: %w", err)
		}
		cfgPath = serveConfigPath
		logger.Info("config loaded", "path", serveConfigPath)
	} else {
		// Try standard locations
		cfgPath = config.FindConfig()
		if cfgPath != "" {
			var err error
			cfg, err = config.Load(cfgPath)
			if err != nil {
				return fmt.Errorf("load config %s: %w", cfgPath, err)
			}
			logger.Info("config found", "path", cfgPath)
		} else {
			cfg = config.DefaultConfig()
			logger.Info("no config file found, using defaults")
		}
	}

	// CLI flags override config
	if serveBind != "" {
		cfg.Bind = serveBind
	}
	if serveMCP {
		if cfg.MCP == nil {
			cfg.MCP = &config.MCPConfig{}
		}
		cfg.MCP.Enabled = true
		if cfg.MCP.Transport == "" {
			cfg.MCP.Transport = "stdio"
		}
		if cfg.MCP.Transport == "sse" && cfg.MCP.Bind == "" {
			cfg.MCP.Bind = "127.0.0.1:11436"
		}
	}

	// llama-server path from persistent flag or env
	llamaServer, err := resolveLlamaServerPath(cmd, cfg)
	if err != nil {
		return err
	}
	cfg.LlamaServer = llamaServer

	// Verify model_dir exists
	if cfg.ModelDir != "" {
		if _, err := os.Stat(cfg.ModelDir); os.IsNotExist(err) {
			logger.Warn("model_dir does not exist, creating", "path", cfg.ModelDir)
			if err := os.MkdirAll(cfg.ModelDir, 0755); err != nil {
				return fmt.Errorf("create model_dir %s: %w", cfg.ModelDir, err)
			}
		}
	}

	srv := server.NewServer(cfg, cfgPath, logger)
	return srv.Run(context.Background())
}
