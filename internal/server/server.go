package server

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/hirdforge/nollama/internal/config"
	"github.com/hirdforge/nollama/internal/hardware"
	"github.com/hirdforge/nollama/internal/process"
)

// Server is the nollama serve daemon.
type Server struct {
	cfg        *config.Config
	cfgPath    string // path to config file (for SIGHUP reload)
	proxy      *Proxy
	procMgr    *process.Manager
	httpServer *http.Server
	logger     *slog.Logger

	mu      sync.Mutex
	running bool
}

// NewServer creates a new serve daemon.
func NewServer(cfg *config.Config, cfgPath string, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	return &Server{
		cfg:     cfg,
		cfgPath: cfgPath,
		proxy:   NewProxy(logger.With("component", "proxy")),
		procMgr: process.NewManager(logger.With("component", "process")),
		logger:  logger,
	}
}

// Run starts the daemon. Blocks until shutdown signal received.
func (s *Server) Run(ctx context.Context) error {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return fmt.Errorf("server already running")
	}
	s.running = true
	s.mu.Unlock()

	s.logger.Info("nollama serve starting",
		"bind", s.cfg.Bind,
		"model_dir", s.cfg.ModelDir,
	)

	// Detect hardware
	hw := s.detectHardware()

	// Set up aliases
	if s.cfg.Aliases != nil {
		s.proxy.SetAliases(s.cfg.Aliases)
	}

	// Autoload models
	if len(s.cfg.Autoload) > 0 {
		s.autoloadModels(hw)
	}

	// Start HTTP server
	s.httpServer = &http.Server{
		Addr:    s.cfg.Bind,
		Handler: s.proxy,
	}

	// Listen first so we can report the actual address
	ln, err := net.Listen("tcp", s.cfg.Bind)
	if err != nil {
		return fmt.Errorf("listen %s: %w", s.cfg.Bind, err)
	}
	s.logger.Info("listening", "addr", ln.Addr().String())

	// Signal handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT, syscall.SIGHUP)

	// Start serving in background
	errCh := make(chan error, 1)
	go func() {
		if err := s.httpServer.Serve(ln); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	s.logger.Info("nollama serve ready",
		"models_loaded", s.proxy.RouteCount(),
	)

	// Wait for signal or context cancellation
	select {
	case sig := <-sigCh:
		switch sig {
		case syscall.SIGHUP:
			s.logger.Info("SIGHUP received, reloading config")
			if err := s.reload(); err != nil {
				s.logger.Error("config reload failed", "error", err)
			}
			// After reload, continue waiting
			return s.waitForShutdown(sigCh, errCh)
		default:
			s.logger.Info("shutdown signal received", "signal", sig)
			return s.shutdown()
		}
	case err := <-errCh:
		return fmt.Errorf("http server error: %w", err)
	case <-ctx.Done():
		s.logger.Info("context cancelled, shutting down")
		return s.shutdown()
	}
}

// waitForShutdown continues the signal loop after a SIGHUP reload.
func (s *Server) waitForShutdown(sigCh chan os.Signal, errCh chan error) error {
	for {
		select {
		case sig := <-sigCh:
			switch sig {
			case syscall.SIGHUP:
				s.logger.Info("SIGHUP received, reloading config")
				if err := s.reload(); err != nil {
					s.logger.Error("config reload failed", "error", err)
				}
				continue
			default:
				s.logger.Info("shutdown signal received", "signal", sig)
				return s.shutdown()
			}
		case err := <-errCh:
			return fmt.Errorf("http server error: %w", err)
		}
	}
}

// shutdown gracefully stops everything.
func (s *Server) shutdown() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.running = false

	// Stop HTTP server
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if s.httpServer != nil {
		s.httpServer.Shutdown(ctx)
	}

	// Stop all llama-server processes
	s.stopAllProcesses()

	s.logger.Info("nollama serve stopped")
	return nil
}

// reload re-reads the config file and reconciles state.
func (s *Server) reload() error {
	if s.cfgPath == "" {
		return fmt.Errorf("no config file to reload (started without --config)")
	}

	newCfg, err := config.Load(s.cfgPath)
	if err != nil {
		return fmt.Errorf("reload config: %w", err)
	}

	s.mu.Lock()
	s.cfg = newCfg
	s.mu.Unlock()

	// Update aliases
	if newCfg.Aliases != nil {
		s.proxy.SetAliases(newCfg.Aliases)
	}

	// Reconcile autoloaded models
	hw := s.detectHardware()
	s.reconcileModels(hw)

	s.logger.Info("config reloaded",
		"models_loaded", s.proxy.RouteCount(),
	)
	return nil
}

// autoloadModels loads all models from the autoload config section.
func (s *Server) autoloadModels(hw *hardware.Inventory) {
	for i, entry := range s.cfg.Autoload {
		modelPath := s.cfg.ModelPath(entry.Model)

		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			s.logger.Error("autoload model not found",
				"model", entry.Model,
				"path", modelPath,
			)
			continue
		}

		s.logger.Info("autoloading model",
			"model", entry.Model,
			"index", i+1,
			"total", len(s.cfg.Autoload),
		)

		port, err := s.loadModel(entry, hw)
		if err != nil {
			s.logger.Error("autoload failed",
				"model", entry.Model,
				"error", err,
			)
			continue
		}

		s.proxy.AddRoute(entry.Model, port)
		s.logger.Info("autoloaded",
			"model", entry.Model,
			"port", port,
		)
	}
}

// loadModel spawns a llama-server process for an autoload entry.
// Returns the port the process is listening on.
func (s *Server) loadModel(entry config.AutoloadEntry, hw *hardware.Inventory) (int, error) {
	modelPath := s.cfg.ModelPath(entry.Model)

	// Build the StartOpts from the autoload entry
	opts := process.StartOpts{
		ModelPath:   modelPath,
		LlamaServer: s.cfg.LlamaServer,
	}

	// GPU selection
	if entry.Device == "cpu" {
		opts.ForceCPU = true
	} else if entry.GPU != nil {
		opts.GPU = *entry.GPU
	}

	// Merge flags: global defaults + per-model
	merged := s.cfg.MergedFlags(entry)
	opts.ExtraFlags = flagsMapToSlice(merged)

	// Hardware for smart defaults
	if hw != nil {
		opts.Hardware = hw
	}

	return s.procMgr.StartOptsStart(opts)
}

// reconcileModels compares desired state (config autoload) with actual state
// (running processes) and loads/unloads as needed.
func (s *Server) reconcileModels(hw *hardware.Inventory) {
	// Build desired set from config
	desired := make(map[string]config.AutoloadEntry)
	for _, entry := range s.cfg.Autoload {
		stem := strings.ToLower(strings.TrimSuffix(entry.Model, ".gguf"))
		desired[stem] = entry
	}

	// Get current running processes
	running := s.procMgr.List()
	currentStems := make(map[string]bool)
	for _, proc := range running {
		stem := strings.ToLower(strings.TrimSuffix(filepath.Base(proc.ModelName), ".gguf"))
		currentStems[stem] = true

		// Unload models no longer in config
		if _, want := desired[stem]; !want {
			s.logger.Info("reconcile: unloading removed model", "model", proc.ModelName)
			s.procMgr.StopByPort(proc.Port)
			s.proxy.RemoveRoute(filepath.Base(proc.ModelName))
		}
	}

	// Load models that should be running but aren't
	for stem, entry := range desired {
		if !currentStems[stem] {
			s.logger.Info("reconcile: loading new model", "model", entry.Model)
			port, err := s.loadModel(entry, hw)
			if err != nil {
				s.logger.Error("reconcile: load failed", "model", entry.Model, "error", err)
				continue
			}
			s.proxy.AddRoute(entry.Model, port)
		}
	}
}

// stopAllProcesses kills every running llama-server.
func (s *Server) stopAllProcesses() {
	procs := s.procMgr.List()
	for _, proc := range procs {
		s.logger.Info("stopping model", "model", proc.ModelName, "pid", proc.PID)
		if _, err := s.procMgr.StopByPort(proc.Port); err != nil {
			s.logger.Error("stop failed", "model", proc.ModelName, "error", err)
		}
	}
}

// detectHardware runs GPU and CPU detection.
func (s *Server) detectHardware() *hardware.Inventory {
	inv, err := hardware.Detect()
	if err != nil {
		s.logger.Warn("hardware detection failed", "error", err)
		return nil
	}

	if inv != nil {
		gpuCount := 0
		if inv.GPUs != nil {
			gpuCount = len(inv.GPUs)
		}
		s.logger.Info("hardware detected",
			"gpus", gpuCount,
			"ram_gb", inv.TotalRAMGB(),
		)
	}
	return inv
}

// flagsMapToSlice converts a map of flag names to values into a string slice.
// e.g. {"ctx-size": 131072, "flash-attn": "on"} → ["--ctx-size", "131072", "--flash-attn", "on"]
func flagsMapToSlice(flags map[string]interface{}) []string {
	var result []string
	for k, v := range flags {
		flag := fmt.Sprintf("--%s", k)
		switch val := v.(type) {
		case bool:
			if val {
				result = append(result, flag)
			}
		case string:
			result = append(result, flag, val)
		default:
			result = append(result, flag, fmt.Sprintf("%v", val))
		}
	}
	return result
}
