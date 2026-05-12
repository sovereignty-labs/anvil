// Package process manages llama-server child processes spawned by nollama.
package process

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

// ProcessInfo holds metadata for a tracked llama-server process.
type ProcessInfo struct {
	PID       int       // OS process ID
	ModelPath string    // absolute path to the GGUF model
	ModelName string    // human-readable model name from GGUF metadata
	Port      int       // llama-server HTTP port
	GPUIndex  string    // e.g. "cuda:0", "cpu"
	StartTime time.Time // when the process was started
	Flags     []string  // the computed (and merged) flags used to launch
	cmd       *exec.Cmd // the underlying exec.Cmd (for stopping)
	logFile   *os.File  // where stdout/stderr are written
}

// ProcessStatus reports whether a ProcessInfo's process is running.
type ProcessStatus string

const (
	ProcessRunning ProcessStatus = "running"
	ProcessStopped ProcessStatus = "stopped"
)

// Status returns the current running/stopped state of the process.
func (p *ProcessInfo) Status() ProcessStatus {
	if p.cmd == nil || p.cmd.Process == nil {
		return ProcessStopped
	}
	err := p.cmd.Process.Signal(syscall.Signal(0))
	if err != nil {
		return ProcessStopped
	}
	return ProcessRunning
}

// Uptime returns how long the process has been running.
func (p *ProcessInfo) Uptime() time.Duration {
	return time.Since(p.StartTime)
}

// Manager tracks all llama-server child processes.
type Manager struct {
	mu       sync.RWMutex
	procs    map[int]*ProcessInfo // keyed by PID
	portMap  map[int]*ProcessInfo // keyed by port
	logDir   string
}

var defaultManager *Manager

func init() {
	defaultManager = &Manager{
		procs:   make(map[int]*ProcessInfo),
		portMap: make(map[int]*ProcessInfo),
		logDir:  "/tmp/nollama",
	}
}

// GetManager returns the singleton Manager instance.
func GetManager() *Manager {
	return defaultManager
}

// NewManager creates a fresh Manager (useful for tests).
func NewManager() *Manager {
	return &Manager{
		procs:   make(map[int]*ProcessInfo),
		portMap: make(map[int]*ProcessInfo),
		logDir:  "/tmp/nollama",
	}
}

// SetLogDir configures where log files are written.
func (m *Manager) SetLogDir(dir string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.logDir = dir
}

// LogDir returns the current log directory.
func (m *Manager) LogDir() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.logDir
}

// ensureLogDir creates the log directory if it doesn't exist.
// Must be called without holding the mutex (reads m.logDir which is set once via SetLogDir).
func (m *Manager) ensureLogDir() error {
	dir := m.logDir
	return os.MkdirAll(dir, 0o755)
}

// openLogFile opens (or creates) a log file for the given port.
// Must be called with m.mu held.
func (m *Manager) openLogFile(port int) (*os.File, error) {
	if err := os.MkdirAll(m.logDir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %w", err)
	}
	name := fmt.Sprintf("llama-server-%d.log", port)
	return os.OpenFile(filepath.Join(m.logDir, name), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
}

// MergePassthroughFlags merges passthrough flags into the computed flags.
// Passthrough flags override computed ones with the same key.
func MergePassthroughFlags(computed []string, passthrough []string) []string {
	if len(passthrough) == 0 {
		return computed
	}

	// Build a set of flag keys that appear in passthrough
	passthroughKeys := make(map[string]bool)
	for i := 0; i < len(passthrough); i++ {
		flag := passthrough[i]
		if strings.HasPrefix(flag, "--") {
			passthroughKeys[flag] = true
			// If the next arg is a value (not a flag), skip it
			if i+1 < len(passthrough) && !strings.HasPrefix(passthrough[i+1], "--") {
				i++
			}
		}
	}

	// Filter out computed flags whose keys are overridden by passthrough
	filtered := make([]string, 0, len(computed))
	for i := 0; i < len(computed); i++ {
		flag := computed[i]
		if strings.HasPrefix(flag, "--") {
			if passthroughKeys[flag] {
				// Skip this flag and its value from computed
				if i+1 < len(computed) && !strings.HasPrefix(computed[i+1], "--") {
					i++
				}
				continue
			}
		}
		filtered = append(filtered, flag)
	}

	// Append passthrough flags
	result := make([]string, 0, len(filtered)+len(passthrough))
	result = append(result, filtered...)
	result = append(result, passthrough...)
	return result
}

// Start spawns a llama-server process from a ComputeFlags Result.
// It captures stdout/stderr to log files in the manager's log directory.
func (m *Manager) Start(result *Result, modelName string, passthrough []string) (*ProcessInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if err := m.ensureLogDir(); err != nil {
		return nil, fmt.Errorf("failed to prepare log directory: %w", err)
	}

	// Merge passthrough flags on top of computed flags
	flags := MergePassthroughFlags(result.Flags, passthrough)

	// Parse the llama-server binary path from the original command
	parts := strings.Fields(result.Command)
	llamaServerPath := parts[0]

	// Open log file
	logFile, err := m.openLogFile(result.Port)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %w", err)
	}

	// Build the actual command
	cmd := exec.Command(llamaServerPath, flags...)

	cmd.Stdout = logFile
	cmd.Stderr = logFile

	// Extract GPU index from SelectedDevice
	gpuIndex := "cpu"
	if !result.CPUFallback {
		gpuIndex = result.SelectedDevice
	}

	procInfo := &ProcessInfo{
		PID:       -1,
		ModelPath: "", // Will be set by the caller
		ModelName: modelName,
		Port:      result.Port,
		GPUIndex:  gpuIndex,
		StartTime: time.Now(),
		Flags:     flags,
		cmd:       cmd,
		logFile:   logFile,
	}

	// Start the process
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return nil, fmt.Errorf("failed to start llama-server: %w", err)
	}

	procInfo.PID = cmd.Process.Pid

	// Track the process
	m.procs[procInfo.PID] = procInfo
	m.portMap[procInfo.Port] = procInfo

	return procInfo, nil
}

// StopByPort stops a process by its port number.
func (m *Manager) StopByPort(port int) (*ProcessInfo, error) {
	m.mu.Lock()
	proc := m.portMap[port]
	m.mu.Unlock()

	if proc == nil {
		return nil, fmt.Errorf("no process found on port %d", port)
	}
	return m.stopProcess(proc), nil
}

// StopByModelName stops a process by its model name.
func (m *Manager) StopByModelName(modelName string) ([]*ProcessInfo, error) {
	m.mu.Lock()
	var matches []*ProcessInfo
	for _, proc := range m.procs {
		if strings.HasSuffix(proc.ModelName, modelName) || strings.HasSuffix(proc.ModelPath, modelName) {
			matches = append(matches, proc)
		}
	}
	m.mu.Unlock()

	if len(matches) == 0 {
		return nil, fmt.Errorf("no running process found for model %q", modelName)
	}

	var stopped []*ProcessInfo
	for _, proc := range matches {
		stopped = append(stopped, m.stopProcess(proc))
	}
	return stopped, nil
}

// Stop finds a process by either port or model name and stops it.
// If port is non-zero, it takes precedence.
func (m *Manager) Stop(port int, modelName string) (*ProcessInfo, error) {
	if port != 0 {
		return m.StopByPort(port)
	}
	if modelName != "" {
		stopped, err := m.StopByModelName(modelName)
		if err != nil {
			return nil, err
		}
		return stopped[0], nil
	}
	return nil, fmt.Errorf("stop requires either a port or model name")
}

// stopProcess sends SIGTERM, waits up to 1s, then SIGKILL if needed.
// Must be called without holding the mutex, or with m.mu as a write lock already held.
func (m *Manager) stopProcess(proc *ProcessInfo) *ProcessInfo {
	if proc.cmd == nil || proc.cmd.Process == nil {
		m.cleanupProcess(proc)
		return proc
	}

	// Send SIGTERM
	if err := proc.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		if err == syscall.ESRCH {
			// Process already gone
			m.cleanupProcess(proc)
			return proc
		}
		// If SIGTERM fails, try SIGKILL immediately
		if err := proc.cmd.Process.Kill(); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to kill process %d: %v\n", proc.PID, err)
		}
		m.cleanupProcess(proc)
		return proc
	}

	// Wait up to 1 second for graceful shutdown
	done := make(chan error, 1)
	go func() {
		done <- proc.cmd.Wait()
	}()

	select {
	case <-done:
		// Process exited gracefully
	case <-time.After(1 * time.Second):
		// Force kill
		proc.cmd.Process.Kill()
		<-done
	}

	m.cleanupProcess(proc)
	return proc
}

// cleanupProcess removes the process from tracking maps and closes the log file.
func (m *Manager) cleanupProcess(proc *ProcessInfo) {
	m.mu.Lock()
	delete(m.procs, proc.PID)
	delete(m.portMap, proc.Port)
	m.mu.Unlock()
	if proc.logFile != nil {
		proc.logFile.Close()
		proc.logFile = nil
	}
}

// List returns all tracked processes with their current status.
func (m *Manager) List() []*ProcessInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]*ProcessInfo, 0, len(m.procs))
	for _, proc := range m.procs {
		result = append(result, proc)
	}

	// Sort by start time for consistent output
	sort.Slice(result, func(i, j int) bool {
		return result[i].StartTime.Before(result[j].StartTime)
	})

	return result
}

// GetByPort returns a process by port, or nil if not found.
func (m *Manager) GetByPort(port int) *ProcessInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.portMap[port]
}

// GetByPID returns a process by PID, or nil if not found.
func (m *Manager) GetByPID(pid int) *ProcessInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.procs[pid]
}

// FormatDuration formats a duration for display (e.g. "2h15m30s").
func FormatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.0fs", d.Seconds())
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	h := d / time.Hour
	d %= time.Hour
	return fmt.Sprintf("%dh%dm%ds", h, int(d.Minutes()), int(d.Seconds())%60)
}

// EndpointURL returns the full endpoint URL for a process.
func EndpointURL(proc *ProcessInfo) string {
	return fmt.Sprintf("http://localhost:%d/v1", proc.Port)
}

// ProcessEndpointURL returns the endpoint URL for a given port.
func ProcessEndpointURL(port int) string {
	return fmt.Sprintf("http://localhost:%d/v1", port)
}
