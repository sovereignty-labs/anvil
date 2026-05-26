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
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/sovereignty-labs/nollama/internal/config"
	"github.com/sovereignty-labs/nollama/internal/hardware"
	nollamamcp "github.com/sovereignty-labs/nollama/internal/mcp"
	"github.com/sovereignty-labs/nollama/internal/process"
	runtimemgr "github.com/sovereignty-labs/nollama/internal/runtime"
)

// Server is the nollama serve daemon.
type Server struct {
	cfg        *config.Config
	cfgPath    string // path to config file (for SIGHUP reload)
	proxy      *Proxy
	procMgr    *process.Manager
	httpServer *http.Server
	mcpRunner  *nollamamcp.Runner
	logger     *slog.Logger

	// stopProcessByPort is the hook the idle reaper calls to terminate a
	// llama-server process. Defaults to procMgr.StopByPort; tests override.
	stopProcessByPort func(int) (*process.ProcessInfo, error)

	mu      sync.Mutex
	running bool
}

// NewServer creates a new serve daemon.
func NewServer(cfg *config.Config, cfgPath string, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	srv := &Server{
		cfg:     cfg,
		cfgPath: cfgPath,
		proxy:   NewProxy(logger.With("component", "proxy")),
		procMgr: process.NewManager(logger.With("component", "process")),
		logger:  logger,
	}
	srv.stopProcessByPort = srv.procMgr.StopByPort
	return srv
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

	// Register signal handlers before any long startup work so SIGHUP is
	// intercepted even while the daemon is still booting.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT, syscall.SIGHUP)
	defer signal.Stop(sigCh)

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
	mux := http.NewServeMux()
	mux.HandleFunc("/api/load", s.handleLoad)
	mux.HandleFunc("/api/unload", s.handleUnload)
	mux.HandleFunc("/api/models", s.handleModels)
	mux.HandleFunc("/api/upload", s.handleUpload)
	mux.HandleFunc("/api/pull", s.handlePull)
	mux.HandleFunc("/api/rm", s.handleRm)
	mux.HandleFunc("/api/status", s.handleStatus)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.Handle("/", s.proxy)
	s.httpServer = &http.Server{
		Addr:    s.cfg.Bind,
		Handler: mux,
	}

	if s.cfg != nil && s.cfg.MCP != nil && s.cfg.MCP.Enabled {
		s.mcpRunner = nollamamcp.NewRunner(s.cfg, "")
		if err := s.mcpRunner.Start(ctx); err != nil {
			return fmt.Errorf("start MCP server: %w", err)
		}
		s.logger.Info("MCP enabled",
			"transport", s.cfg.MCP.Transport,
			"bind", s.cfg.MCP.Bind,
		)
	}

	// Listen first so we can report the actual address
	ln, err := net.Listen("tcp", s.cfg.Bind)
	if err != nil {
		return fmt.Errorf("listen %s: %w", s.cfg.Bind, err)
	}
	s.logger.Info("listening", "addr", ln.Addr().String())

	// Start serving in background
	errCh := make(chan error, 1)
	go func() {
		if err := s.httpServer.Serve(ln); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	// Idle reaper — proactively unloads models idle longer than swap.idle_timeout.
	reaperCtx, cancelReaper := context.WithCancel(ctx)
	defer cancelReaper()
	go s.startIdleReaper(reaperCtx)

	s.logger.Info("nollama serve ready",
		"models_loaded", s.proxy.RouteCount(),
	)

	// Wait for signal or context cancellation
	for {
		select {
		case sig := <-sigCh:
			switch sig {
			case syscall.SIGHUP:
				s.logger.Info("SIGHUP received, reloading config")
				if err := s.reloadConfig(); err != nil {
					s.logger.Error("config reload failed", "error", err)
				}
				continue
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

	if s.mcpRunner != nil {
		_ = s.mcpRunner.Shutdown(ctx)
	}

	// Stop all llama-server processes
	s.stopAllProcesses()

	s.logger.Info("nollama serve stopped")
	return nil
}

// reloadConfig re-reads the config file and reconciles state.
func (s *Server) reloadConfig() error {
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

	// Update MCP runner config
	if s.mcpRunner != nil {
		s.mcpRunner.UpdateConfig(newCfg)
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

		s.proxy.AddRouteWithAlias(entry.Model, port, entry.Alias)
		s.logger.Info("autoloaded",
			"model", entry.Model,
			"alias", entry.Alias,
			"port", port,
		)
	}
}

// loadModel spawns a llama-server process for an autoload entry.
// Returns the port the process is listening on.
func (s *Server) loadModel(entry config.AutoloadEntry, hw *hardware.Inventory) (int, error) {
	modelPath := s.cfg.ModelPath(entry.Model)
	_, backend, err := s.resolveLlamaServerPathAndBackend()
	if err != nil {
		return 0, err
	}

	// Build the StartOpts from the autoload entry
	opts := process.StartOpts{
		ModelPath:   modelPath,
		LlamaServer: s.cfg.LlamaServer,
		Backend:     backend,
		PinnedPort:  entry.Port,
	}

	// GPU selection
	if entry.Device == "cpu" {
		opts.ForceCPU = true
	} else if entry.GPU != nil {
		opts.GPU = *entry.GPU
	}

	// Merge flags: defaults + profiles + per-model
	merged, requires, err := s.cfg.MergedFlagsWithProfiles(entry)
	if err != nil {
		return 0, err
	}
	for _, warning := range s.profileWarnings(requires) {
		s.logger.Warn("autoload profile requirement mismatch",
			"model", entry.Model,
			"warning", warning,
		)
	}
	opts.Env, merged = buildAutoloadEnv(entry, merged)
	opts.ExtraFlags = config.FlagsMapToSlice(merged)
	// When an alias is configured, pass --alias to llama-server so its
	// /v1/models response advertises the alias as the model name.
	if alias := strings.TrimSpace(entry.Alias); alias != "" {
		opts.ExtraFlags = append(opts.ExtraFlags, "--alias", alias)
	}

	// Hardware for smart defaults
	if hw != nil {
		opts.Hardware = hw
	}

	// Pass nollama's bind port so the process manager can avoid it
	_, portStr, err := net.SplitHostPort(s.cfg.Bind)
	if err == nil {
		var portNum int
		if portNum, err = strconv.Atoi(portStr); err == nil {
			opts.ReservedPort = portNum
		}
	}
	if err != nil {
		s.logger.Warn("failed to parse bind port from config", "bind", s.cfg.Bind, "error", err)
	}

	return s.procMgr.StartOptsStart(opts)
}

func buildAutoloadEnv(entry config.AutoloadEntry, merged map[string]interface{}) (map[string]string, map[string]interface{}) {
	if len(entry.Env) == 0 && len(merged) == 0 {
		return nil, merged
	}

	env := make(map[string]string, len(entry.Env)+1)
	for key, value := range entry.Env {
		env[key] = value
	}
	if raw, ok := merged["vk-device"]; ok {
		env["GGML_VK_DEVICE"] = fmt.Sprintf("%v", raw)
		delete(merged, "vk-device")
	}
	if len(env) == 0 {
		env = nil
	}
	return env, merged
}

func (s *Server) profileWarnings(requires []config.ProfileRequires) []string {
	// When llama_server is set explicitly the user is managing the binary path
	// directly, so profile runtime requirements are moot — suppress the warning.
	if s.cfg != nil && strings.TrimSpace(s.cfg.LlamaServer) != "" {
		return nil
	}
	activeRuntime, err := runtimemgr.NewManager().ActiveName()
	if err != nil {
		activeRuntime = ""
	}
	return config.ProfileRuntimeWarnings(requires, activeRuntime)
}

// reconcileModels compares desired state (config autoload) with actual state
// (running processes) and loads/unloads as needed.
func (s *Server) reconcileModels(hw *hardware.Inventory) {
	// Build desired set from config keyed on the proxy route key so multiple
	// autoload entries that share a filename but differ by alias coexist.
	desired := make(map[string]config.AutoloadEntry)
	for _, entry := range s.cfg.Autoload {
		desired[routeKeyFor(entry.Model, entry.Alias)] = entry
	}

	// Cross-reference running processes against current proxy routes so we
	// can identify each process by route key (which carries the alias).
	routeKeyByPort := make(map[int]string)
	for _, r := range s.proxy.RouteStatsList() {
		routeKeyByPort[r.Port] = r.RouteKey
	}

	running := s.procMgr.List()
	currentKeys := make(map[string]bool)
	for _, proc := range running {
		key, ok := routeKeyByPort[proc.Port]
		if !ok {
			// Process is running but no proxy route exists for it — fall back to
			// filename stem so old entries still get cleaned up.
			key = routeKeyFor(filepath.Base(proc.ModelName), "")
		}
		currentKeys[key] = true

		if _, want := desired[key]; !want {
			s.logger.Info("reconcile: unloading removed model", "model", proc.ModelName, "key", key)
			s.procMgr.StopByPort(proc.Port)
			s.proxy.RemoveRouteByPort(proc.Port)
		}
	}

	for key, entry := range desired {
		if !currentKeys[key] {
			s.logger.Info("reconcile: loading new model", "model", entry.Model, "alias", entry.Alias)
			port, err := s.loadModel(entry, hw)
			if err != nil {
				s.logger.Error("reconcile: load failed", "model", entry.Model, "error", err)
				continue
			}
			s.proxy.AddRouteWithAlias(entry.Model, port, entry.Alias)
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
