package server

import (
	"log/slog"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sovereignty-labs/anvil/internal/config"
	"github.com/sovereignty-labs/anvil/internal/process"
)

func newReaperTestServer(t *testing.T, idleTimeout string) (*Server, *int32) {
	t.Helper()
	cfg := &config.Config{
		Swap: &config.SwapConfig{IdleTimeout: idleTimeout},
	}
	s := &Server{
		cfg:    cfg,
		proxy:  NewProxy(slog.Default()),
		logger: slog.Default(),
	}
	var calls int32
	s.stopProcessByPort = func(port int) (*process.ProcessInfo, error) {
		atomic.AddInt32(&calls, 1)
		return nil, nil
	}
	return s, &calls
}

func TestReapIdleModelsUnloadsIdleModel(t *testing.T) {
	s, calls := newReaperTestServer(t, "30m")
	s.proxy.AddRoute("idle.gguf", 11111)

	// Age the route past the threshold and grace period.
	s.proxy.mu.Lock()
	for _, r := range s.proxy.all {
		r.LoadedAt = time.Now().Add(-2 * time.Hour)
		r.LastRequest = time.Now().Add(-2 * time.Hour)
	}
	s.proxy.mu.Unlock()

	s.reapIdleModels()

	if got := atomic.LoadInt32(calls); got != 1 {
		t.Errorf("expected 1 stopProcessByPort call, got %d", got)
	}
	if s.proxy.RouteCount() != 0 {
		t.Errorf("expected route to be removed, %d remain", s.proxy.RouteCount())
	}
}

func TestReapIdleModelsDisabledWhenZeroTimeout(t *testing.T) {
	s, calls := newReaperTestServer(t, "")
	s.proxy.AddRoute("idle.gguf", 11111)

	s.proxy.mu.Lock()
	for _, r := range s.proxy.all {
		r.LoadedAt = time.Now().Add(-24 * time.Hour)
		r.LastRequest = time.Now().Add(-24 * time.Hour)
	}
	s.proxy.mu.Unlock()

	s.reapIdleModels()

	if got := atomic.LoadInt32(calls); got != 0 {
		t.Errorf("expected 0 stop calls when idle_timeout unset, got %d", got)
	}
	if s.proxy.RouteCount() != 1 {
		t.Errorf("expected route preserved, got %d routes", s.proxy.RouteCount())
	}
}

func TestReapIdleModelsRespectsGracePeriod(t *testing.T) {
	s, calls := newReaperTestServer(t, "30m")
	s.proxy.AddRoute("fresh.gguf", 11111)

	// LastRequest is ancient, but LoadedAt is recent (just added).
	s.proxy.mu.Lock()
	for _, r := range s.proxy.all {
		r.LastRequest = time.Now().Add(-24 * time.Hour)
	}
	s.proxy.mu.Unlock()

	s.reapIdleModels()

	if got := atomic.LoadInt32(calls); got != 0 {
		t.Errorf("expected 0 stops within grace period, got %d", got)
	}
	if s.proxy.RouteCount() != 1 {
		t.Errorf("expected route preserved within grace period, got %d", s.proxy.RouteCount())
	}
}

func TestReapIdleModelsStopFailureLeavesRoute(t *testing.T) {
	s, _ := newReaperTestServer(t, "30m")
	s.stopProcessByPort = func(port int) (*process.ProcessInfo, error) {
		return nil, errStopFailed
	}
	s.proxy.AddRoute("idle.gguf", 11111)
	s.proxy.mu.Lock()
	for _, r := range s.proxy.all {
		r.LoadedAt = time.Now().Add(-2 * time.Hour)
		r.LastRequest = time.Now().Add(-2 * time.Hour)
	}
	s.proxy.mu.Unlock()

	s.reapIdleModels()

	if s.proxy.RouteCount() != 1 {
		t.Errorf("expected route preserved when stop fails, got %d", s.proxy.RouteCount())
	}
}

var errStopFailed = stopErr("stop failed")

type stopErr string

func (e stopErr) Error() string { return string(e) }
