package server

import (
	"context"
	"time"

	"github.com/hirdforge/nollama/internal/config"
)

// idleReaperTickInterval is how often the reaper wakes up to scan for idle
// routes. The idle threshold itself is configurable; the cadence is not.
const idleReaperTickInterval = 60 * time.Second

// startIdleReaper runs the idle-unload loop until ctx is cancelled. On every
// tick it re-reads swap.idle_timeout from the live config so SIGHUP changes
// take effect on the next iteration.
func (s *Server) startIdleReaper(ctx context.Context) {
	ticker := time.NewTicker(idleReaperTickInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.reapIdleModels()
		}
	}
}

// reapIdleModels scans routes for ones idle longer than swap.idle_timeout and
// unloads them. No-op when idle_timeout is zero, unset, or unparseable.
func (s *Server) reapIdleModels() {
	cfg := s.snapshotConfig()
	threshold := swapIdleTimeout(cfg)
	if threshold <= 0 {
		return
	}

	idle := s.proxy.IdleRoutes(threshold)
	for _, r := range idle {
		s.unloadIdleRoute(r, threshold)
	}
}

// unloadIdleRoute stops the backing process and removes the proxy route.
func (s *Server) unloadIdleRoute(r IdleRoute, threshold time.Duration) {
	idleFor := time.Since(r.IdleSince).Round(time.Second)
	s.logger.Info("auto-unload",
		"model", r.ModelName,
		"idle", idleFor,
		"threshold", threshold,
	)

	if _, err := s.stopProcessByPort(r.Port); err != nil {
		s.logger.Error("auto-unload: stop process failed",
			"model", r.ModelName,
			"port", r.Port,
			"error", err,
		)
		return
	}
	s.proxy.RemoveRouteByPort(r.Port)
}

// snapshotConfig returns the current config under the server mutex.
func (s *Server) snapshotConfig() *config.Config {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cfg
}

func swapIdleTimeout(cfg *config.Config) time.Duration {
	if cfg == nil || cfg.Swap == nil {
		return 0
	}
	return cfg.Swap.IdleTimeoutDuration()
}
