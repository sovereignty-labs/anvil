package mcp

import (
	"context"
	"fmt"
	"os"
	"strings"

	mcpsrv "github.com/mark3labs/mcp-go/server"
	"github.com/sovereignty-labs/nollama/internal/config"
	"github.com/sovereignty-labs/nollama/internal/federation"
	"github.com/sovereignty-labs/nollama/internal/version"
)

// Runner owns the MCP server lifecycle.
type Runner struct {
	cfg          *config.Config
	registryPath string
	localBaseURL string

	server   *mcpsrv.MCPServer
	stdioSrv *mcpsrv.StdioServer
	sseSrv   *mcpsrv.SSEServer
	cancel   context.CancelFunc
}

// NewRunner builds an MCP runner for the provided config.
func NewRunner(cfg *config.Config, registryPath string) *Runner {
	if cfg == nil {
		cfg = config.DefaultConfig()
	}
	if registryPath == "" {
		registryPath = federation.DefaultRegistryPath()
	}

	r := &Runner{
		cfg:          cfg,
		registryPath: registryPath,
		localBaseURL: managementBaseURL(cfg.Bind),
	}
	r.server = mcpsrv.NewMCPServer(
		"nollama",
		version.Version,
		mcpsrv.WithToolCapabilities(true),
	)
	r.registerTools()
	return r
}

// Start launches the configured transport.
func (r *Runner) Start(ctx context.Context) error {
	if r == nil || r.server == nil {
		return nil
	}

	transport := "stdio"
	if r.cfg != nil && r.cfg.MCP != nil && strings.TrimSpace(r.cfg.MCP.Transport) != "" {
		transport = strings.ToLower(strings.TrimSpace(r.cfg.MCP.Transport))
	}

	runCtx, cancel := context.WithCancel(ctx)
	r.cancel = cancel

	switch transport {
	case "stdio":
		r.stdioSrv = mcpsrv.NewStdioServer(r.server)
		go func() {
			_ = r.stdioSrv.Listen(runCtx, os.Stdin, os.Stdout)
		}()
		return nil
	case "sse":
		bind := "127.0.0.1:11436"
		if r.cfg != nil && r.cfg.MCP != nil && strings.TrimSpace(r.cfg.MCP.Bind) != "" {
			bind = strings.TrimSpace(r.cfg.MCP.Bind)
		}
		r.sseSrv = mcpsrv.NewSSEServer(r.server)
		go func() {
			_ = r.sseSrv.Start(bind)
		}()
		go func() {
			<-runCtx.Done()
			_ = r.sseSrv.Shutdown(context.Background())
		}()
		return nil
	default:
		return fmt.Errorf("unsupported MCP transport %q", transport)
	}
}

// UpdateConfig replaces the runner's config (called on SIGHUP reload).
func (r *Runner) UpdateConfig(cfg *config.Config) {
	if r == nil {
		return
	}
	r.cfg = cfg
}

// Shutdown stops the MCP transport.
func (r *Runner) Shutdown(ctx context.Context) error {
	if r == nil {
		return nil
	}
	if r.cancel != nil {
		r.cancel()
	}
	if r.sseSrv != nil {
		return r.sseSrv.Shutdown(ctx)
	}
	return nil
}

func managementBaseURL(bind string) string {
	bind = strings.TrimSpace(bind)
	if bind == "" {
		return "http://127.0.0.1:11434"
	}
	if strings.HasPrefix(bind, "http://") || strings.HasPrefix(bind, "https://") {
		return strings.TrimRight(bind, "/")
	}
	if strings.Contains(bind, ":") {
		return "http://" + bind
	}
	return "http://127.0.0.1:11434"
}
