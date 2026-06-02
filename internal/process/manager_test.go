package process

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	runtimemgr "github.com/sovereignty-labs/anvil/internal/runtime"
)

// makeTestResult creates a minimal Result for testing Start/Stop.
func makeTestResult(t *testing.T, port int, llamaServerPath string, flags ...string) *Result {
	t.Helper()
	allFlags := append([]string{
		"--model", "/tmp/test/model.gguf",
		"--host", "0.0.0.0",
		"--port", fmt.Sprintf("%d", port),
	}, flags...)
	return &Result{
		SelectedDevice: "cuda:0",
		Backend:        runtimemgr.BuildBackendCUDA,
		Flags:          allFlags,
		Command:        llamaServerPath + " " + fmt.Sprintf("--model /tmp/test/model.gguf --host 0.0.0.0 --port %d", port),
		VRAMUsedMB:     4096,
		VRAMTotalMB:    12288,
		CPUFallback:    false,
		GPUIndex:       0,
		Port:           port,
	}
}

func TestManager_Start(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	// Create a mock script that sleeps
	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19999, mockScript)

	proc, err := manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	if proc == nil {
		t.Fatal("expected non-nil process info")
	}

	if proc.PID <= 0 {
		t.Errorf("expected positive PID, got %d", proc.PID)
	}

	if proc.ModelName != "test-model.gguf" {
		t.Errorf("expected model name 'test-model.gguf', got %q", proc.ModelName)
	}

	if proc.Port != 19999 {
		t.Errorf("expected port 19999, got %d", proc.Port)
	}

	if proc.Status() != ProcessRunning {
		t.Errorf("expected status running, got %s", proc.Status())
	}

	// Verify process is alive
	process := manager.GetByPID(proc.PID)
	if process == nil {
		t.Fatal("process not found by PID in manager")
	}
	if process.Status() != ProcessRunning {
		t.Errorf("process not running after retrieval by PID")
	}

	// Verify port map
	portProc := manager.GetByPort(19999)
	if portProc == nil {
		t.Fatal("process not found by port in manager")
	}

	// Cleanup
	manager.StopByPort(19999)
}

func TestManager_Start_MissingBinary(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	result := makeTestResult(t, 19998, "/nonexistent/path/to/llama-server")

	_, err := manager.Start(result, "test-model.gguf", nil)
	if err == nil {
		t.Fatal("expected error when starting with missing binary")
	}
}

func TestManager_Stop(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19997, mockScript)

	proc, err := manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	// Verify it's running
	if proc.Status() != ProcessRunning {
		t.Fatal("expected process to be running before Stop")
	}

	stopped, err := manager.StopByPort(19997)
	if err != nil {
		t.Fatalf("StopByPort() error: %v", err)
	}

	if stopped.PID != proc.PID {
		t.Errorf("expected PID %d, got %d", proc.PID, stopped.PID)
	}

	// Give the process a moment to actually die
	time.Sleep(200 * time.Millisecond)

	// Verify it's stopped
	if stopped.Status() != ProcessStopped {
		t.Errorf("expected process to be stopped after StopByPort, got %s", stopped.Status())
	}

	// Verify it's no longer in the manager maps
	if manager.GetByPID(proc.PID) != nil {
		t.Error("process still found by PID after stop")
	}
	if manager.GetByPort(19997) != nil {
		t.Error("process still found by port after stop")
	}
}

func TestManager_Stop_ModelName(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19996, mockScript)

	proc, err := manager.Start(result, "my-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	stopped, err := manager.StopByModelName("my-model.gguf")
	if err != nil {
		t.Fatalf("StopByModelName() error: %v", err)
	}

	if len(stopped) == 0 {
		t.Fatal("expected at least one stopped process")
	}

	if stopped[0].PID != proc.PID {
		t.Errorf("expected PID %d, got %d", proc.PID, stopped[0].PID)
	}

	// Verify it's stopped
	time.Sleep(200 * time.Millisecond)
	if stopped[0].Status() != ProcessStopped {
		t.Errorf("expected process to be stopped, got %s", stopped[0].Status())
	}
}

func TestManager_Stop_ModelNotFound(t *testing.T) {
	manager := NewManager(nil)

	_, err := manager.StopByModelName("nonexistent-model.gguf")
	if err == nil {
		t.Fatal("expected error when stopping nonexistent model")
	}

	_, err = manager.StopByPort(99999)
	if err == nil {
		t.Fatal("expected error when stopping nonexistent port")
	}
}

func TestManager_List(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result1 := makeTestResult(t, 19990, mockScript)
	result2 := makeTestResult(t, 19991, mockScript)

	proc1, err := manager.Start(result1, "model1.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error for model1: %v", err)
	}
	proc2, err := manager.Start(result2, "model2.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error for model2: %v", err)
	}

	procs := manager.List()

	if len(procs) != 2 {
		t.Errorf("expected 2 processes, got %d", len(procs))
	}

	// Verify both processes are in the list
	pids := make(map[int]bool)
	for _, p := range procs {
		pids[p.PID] = true
	}

	if !pids[proc1.PID] {
		t.Errorf("PID %d not found in List()", proc1.PID)
	}
	if !pids[proc2.PID] {
		t.Errorf("PID %d not found in List()", proc2.PID)
	}

	// Cleanup
	manager.StopByPort(19990)
	manager.StopByPort(19991)
}

func TestManager_List_Empty(t *testing.T) {
	manager := NewManager(nil)
	procs := manager.List()
	if len(procs) != 0 {
		t.Errorf("expected 0 processes, got %d", len(procs))
	}
}

func TestManager_LogFile(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\necho hello from llama-server\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19988, mockScript)

	_, err = manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	// Give the process time to write some output
	time.Sleep(300 * time.Millisecond)

	// Check that the log file exists
	logPath := filepath.Join(tmpDir, "llama-server-19988.log")
	if _, err := os.Stat(logPath); os.IsNotExist(err) {
		t.Errorf("log file not found at %s", logPath)
	} else {
		// Read the log and verify it has content
		content, readErr := os.ReadFile(logPath)
		if readErr != nil {
			t.Errorf("failed to read log file: %v", readErr)
		} else if len(content) == 0 {
			t.Error("log file is empty")
		}
	}

	manager.StopByPort(19988)
}

func TestManager_MultipleProcesses(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	ports := []int{19980, 19981, 19982}
	var procs []*ProcessInfo

	for i, port := range ports {
		result := makeTestResult(t, port, mockScript)
		proc, err := manager.Start(result, fmt.Sprintf("model%d.gguf", i), nil)
		if err != nil {
			t.Fatalf("Start() error for port %d: %v", port, err)
		}
		procs = append(procs, proc)
	}

	if len(manager.List()) != 3 {
		t.Errorf("expected 3 processes, got %d", len(manager.List()))
	}

	// Stop middle one
	manager.StopByPort(19981)

	// Should have 2 remaining
	remaining := manager.List()
	if len(remaining) != 2 {
		t.Errorf("expected 2 remaining processes, got %d", len(remaining))
	}

	// Cleanup
	manager.StopByPort(19980)
	manager.StopByPort(19982)
}

func TestManager_Uptime(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19979, mockScript)
	proc, err := manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	time.Sleep(100 * time.Millisecond)
	uptime := proc.Uptime()
	if uptime < 50*time.Millisecond {
		t.Errorf("expected uptime >= 50ms, got %v", uptime)
	}

	manager.StopByPort(19979)
}

func TestProcessEndpointURL(t *testing.T) {
	if ProcessEndpointURL(8080) != "http://localhost:8080/v1" {
		t.Errorf("unexpected endpoint URL for port 8080")
	}
	if ProcessEndpointURL(11434) != "http://localhost:11434/v1" {
		t.Errorf("unexpected endpoint URL for port 11434")
	}
}

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		input    time.Duration
		expected string
	}{
		{5 * time.Second, "5s"},
		{90 * time.Second, "1m30s"},
		{90 * time.Minute, "1h30m0s"},
		{2*time.Hour + 15*time.Minute + 30*time.Second, "2h15m30s"},
		{3 * time.Second, "3s"},
	}

	for _, tt := range tests {
		result := FormatDuration(tt.input)
		if result != tt.expected {
			t.Errorf("FormatDuration(%v) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestProcessInfo_Status(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19978, mockScript)
	proc, err := manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	if proc.Status() != ProcessRunning {
		t.Errorf("expected running status, got %s", proc.Status())
	}

	manager.StopByPort(19978)
	time.Sleep(200 * time.Millisecond)

	if proc.Status() != ProcessStopped {
		t.Errorf("expected stopped status after stop, got %s", proc.Status())
	}
}

func TestNewManager(t *testing.T) {
	m := NewManager(nil)
	if m == nil {
		t.Fatal("NewManager returned nil")
	}
	if m.procs == nil {
		t.Fatal("NewManager returned manager with nil procs map")
	}
	if m.portMap == nil {
		t.Fatal("NewManager returned manager with nil portMap")
	}
	if m.logDir != "/tmp/anvil" {
		t.Errorf("expected default logDir /tmp/anvil, got %q", m.logDir)
	}
}

func TestManager_GetByPID(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19977, mockScript)
	proc, err := manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	found := manager.GetByPID(proc.PID)
	if found == nil {
		t.Fatal("GetByPID returned nil for known PID")
	}
	if found.PID != proc.PID {
		t.Errorf("GetByPID returned wrong PID")
	}

	notFound := manager.GetByPID(99999)
	if notFound != nil {
		t.Error("GetByPID returned non-nil for unknown PID")
	}

	manager.StopByPort(19977)
}

func TestManager_GetByPort(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19976, mockScript)
	_, err = manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	found := manager.GetByPort(19976)
	if found == nil {
		t.Fatal("GetByPort returned nil for known port")
	}

	notFound := manager.GetByPort(99999)
	if notFound != nil {
		t.Error("GetByPort returned non-nil for unknown port")
	}

	manager.StopByPort(19976)
}

func TestMergePassthroughFlags_NoPassthrough(t *testing.T) {
	computed := []string{"--model", "test.gguf", "--port", "8080", "--ctx-size", "4096"}
	merged := MergePassthroughFlags(computed, nil)
	if len(merged) != len(computed) {
		t.Errorf("expected %d flags, got %d", len(computed), len(merged))
	}
	for i, f := range computed {
		if merged[i] != f {
			t.Errorf("flag mismatch at %d: expected %q, got %q", i, f, merged[i])
		}
	}
}

func TestMergePassthroughFlags_SingleOverride(t *testing.T) {
	computed := []string{"--model", "test.gguf", "--port", "8080", "--ctx-size", "4096"}
	passthrough := []string{"--ctx-size", "8192"}
	merged := MergePassthroughFlags(computed, passthrough)

	// Should have --port 8080, and passthrough --ctx-size 8192 (not computed 4096)
	foundCtx := false
	foundPort := false
	for i, f := range merged {
		if f == "--ctx-size" {
			foundCtx = true
			if i+1 >= len(merged) || merged[i+1] != "8192" {
				t.Errorf("expected --ctx-size value 8192, got %q", merged[i+1])
			}
		}
		if f == "--port" {
			foundPort = true
			if i+1 >= len(merged) || merged[i+1] != "8080" {
				t.Errorf("expected --port value 8080, got %q", merged[i+1])
			}
		}
	}
	if !foundCtx {
		t.Error("expected --ctx-size in merged flags")
	}
	if !foundPort {
		t.Error("expected --port in merged flags")
	}
}

func TestMergePassthroughFlags_MultipleOverrides(t *testing.T) {
	computed := []string{"--model", "test.gguf", "--port", "8080", "--ctx-size", "4096", "--threads", "8"}
	passthrough := []string{"--ctx-size", "8192", "--threads", "16"}
	merged := MergePassthroughFlags(computed, passthrough)

	// --ctx-size and --threads from computed should be removed
	// passthrough values should be added at end
	for i, f := range merged {
		if f == "--ctx-size" && i+1 < len(merged) {
			if merged[i+1] != "8192" {
				t.Errorf("expected ctx-size 8192, got %q", merged[i+1])
			}
		}
		if f == "--threads" && i+1 < len(merged) {
			// Could be from computed (8) or passthrough (16)
			// Since passthrough wins, 16 should appear
		}
	}

	// Count occurrences of --ctx-size (should be 1)
	ctxCount := 0
	for _, f := range merged {
		if f == "--ctx-size" {
			ctxCount++
		}
	}
	if ctxCount != 1 {
		t.Errorf("expected exactly 1 --ctx-size, got %d", ctxCount)
	}

	// Count occurrences of --threads (should be 1)
	threadCount := 0
	for _, f := range merged {
		if f == "--threads" {
			threadCount++
		}
	}
	if threadCount != 1 {
		t.Errorf("expected exactly 1 --threads, got %d", threadCount)
	}
}

func TestMergePassthroughFlags_AddNewFlags(t *testing.T) {
	computed := []string{"--model", "test.gguf", "--port", "8080"}
	passthrough := []string{"--jinja", "--n-predict", "128"}
	merged := MergePassthroughFlags(computed, passthrough)

	// Should have all original flags plus the new passthrough flags
	if len(merged) != 7 {
		t.Errorf("expected 7 flags, got %d: %v", len(merged), merged)
	}

	// Check that passthrough flags are present
	foundJinja := false
	foundNPredict := false
	for i, f := range merged {
		if f == "--jinja" {
			foundJinja = true
		}
		if f == "--n-predict" && i+1 < len(merged) && merged[i+1] == "128" {
			foundNPredict = true
		}
	}
	if !foundJinja {
		t.Error("expected --jinja in merged flags")
	}
	if !foundNPredict {
		t.Error("expected --n-predict 128 in merged flags")
	}
}

func TestMergePassthroughFlags_OverrideAndAdd(t *testing.T) {
	computed := []string{"--model", "test.gguf", "--ctx-size", "4096", "--n-predict", "64"}
	passthrough := []string{"--ctx-size", "8192", "--log-level", "debug"}
	merged := MergePassthroughFlags(computed, passthrough)

	// Should have --model test.gguf (from computed)
	// --ctx-size 8192 (from passthrough, overriding computed)
	// --log-level debug (from passthrough, new)
	// --n-predict 64 should be gone (overridden by ctx-size override... no wait)
	// Actually --ctx-size override means computed --ctx-size 4096 is removed
	// --n-predict 64 is NOT overridden, so it should remain
	// But wait: the passthrough has --ctx-size 8192 and --log-level debug
	// The computed --ctx-size 4096 gets removed, --n-predict 64 stays

	foundModel := false
	foundCtx := false
	foundNPredict := false
	foundLogLevel := false

	for i, f := range merged {
		switch f {
		case "--model":
			foundModel = true
		case "--ctx-size":
			foundCtx = true
			if i+1 < len(merged) && merged[i+1] != "8192" {
				t.Errorf("expected ctx-size 8192, got %q", merged[i+1])
			}
		case "--n-predict":
			foundNPredict = true
			if i+1 < len(merged) && merged[i+1] != "64" {
				t.Errorf("expected n-predict 64, got %q", merged[i+1])
			}
		case "--log-level":
			foundLogLevel = true
			if i+1 < len(merged) && merged[i+1] != "debug" {
				t.Errorf("expected log-level debug, got %q", merged[i+1])
			}
		}
	}

	if !foundModel {
		t.Error("expected --model in merged flags")
	}
	if !foundCtx {
		t.Error("expected --ctx-size in merged flags")
	}
	if !foundNPredict {
		t.Error("expected --n-predict in merged flags")
	}
	if !foundLogLevel {
		t.Error("expected --log-level in merged flags")
	}
}

func TestEndpointURL(t *testing.T) {
	// Create a mock ProcessInfo
	proc := &ProcessInfo{
		PID:  1234,
		Port: 8080,
	}

	url := EndpointURL(proc)
	if url != "http://localhost:8080/v1" {
		t.Errorf("expected http://localhost:8080/v1, got %s", url)
	}
}

func TestManager_Stop_SignalGraceful(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	// Create a mock script that responds to SIGTERM
	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\ntrap 'exit 0' TERM\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19975, mockScript)
	proc, err := manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	// Stop should send SIGTERM and the process should exit gracefully
	_, err = manager.StopByPort(19975)
	if err != nil {
		t.Fatalf("StopByPort() error: %v", err)
	}

	// Give time for graceful shutdown
	time.Sleep(500 * time.Millisecond)

	if proc.Status() != ProcessStopped {
		t.Errorf("expected stopped status, got %s", proc.Status())
	}
}

func TestManager_Stop_KillAfterTimeout(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	// Create a mock script that ignores SIGTERM
	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\ntrap '' TERM\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19974, mockScript)
	proc, err := manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	// Stop should eventually SIGKILL the process
	_, err = manager.StopByPort(19974)
	if err != nil {
		t.Fatalf("StopByPort() error: %v", err)
	}

	// Give time for SIGKILL to take effect
	time.Sleep(1 * time.Second)

	if proc.Status() != ProcessStopped {
		t.Errorf("expected stopped status after SIGKILL, got %s", proc.Status())
	}
}

func TestManager_ConcurrentStartStop(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	const numProcesses = 5
	var wg sync.WaitGroup
	procs := make([]*ProcessInfo, numProcesses)

	// Start processes concurrently
	for i := 0; i < numProcesses; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			port := 19900 + idx
			result := makeTestResult(t, port, mockScript)
			proc, err := manager.Start(result, fmt.Sprintf("model-%d.gguf", idx), nil)
			if err != nil {
				t.Errorf("Start() error for port %d: %v", port, err)
				return
			}
			procs[idx] = proc
		}(i)
	}
	wg.Wait()

	// Verify all started
	running := 0
	for _, p := range procs {
		if p != nil {
			running++
		}
	}
	if running != numProcesses {
		t.Errorf("expected %d running processes, got %d", numProcesses, running)
	}

	// Stop processes concurrently
	for i := 0; i < numProcesses; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			manager.StopByPort(19900 + idx)
		}(i)
	}
	wg.Wait()

	// Give time for cleanup
	time.Sleep(300 * time.Millisecond)

	// Verify all stopped
	procs = manager.List()
	if len(procs) != 0 {
		t.Errorf("expected 0 remaining processes, got %d", len(procs))
	}
}

// Test that the Manager doesn't leak processes on repeated Start/Stop cycles.
func TestManager_Reuse(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	// Start and stop several times
	for i := 0; i < 3; i++ {
		result := makeTestResult(t, 19960, mockScript)
		proc, err := manager.Start(result, "test-model.gguf", nil)
		if err != nil {
			t.Fatalf("Start() iteration %d error: %v", i, err)
		}

		if proc.Status() != ProcessRunning {
			t.Errorf("iteration %d: expected running status", i)
		}

		manager.StopByPort(19960)
		time.Sleep(100 * time.Millisecond)

		if proc.Status() != ProcessStopped {
			t.Errorf("iteration %d: expected stopped status after stop", i)
		}
	}

	if len(manager.List()) != 0 {
		t.Error("expected no remaining processes after clean stop/restart cycle")
	}
}

func TestProcessInfo_Uptime_Increasing(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19959, mockScript)
	proc, err := manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	initialUptime := proc.Uptime()
	time.Sleep(50 * time.Millisecond)
	secondUptime := proc.Uptime()

	if secondUptime <= initialUptime {
		t.Errorf("uptime should increase: %v <= %v", secondUptime, initialUptime)
	}

	manager.StopByPort(19959)
}

// Test that LogDir returns the configured value.
func TestManager_LogDir(t *testing.T) {
	manager := NewManager(nil)
	if manager.LogDir() != "/tmp/anvil" {
		t.Errorf("expected default log dir '/tmp/anvil', got %q", manager.LogDir())
	}

	manager.SetLogDir("/custom/logs")
	if manager.LogDir() != "/custom/logs" {
		t.Errorf("expected custom log dir '/custom/logs', got %q", manager.LogDir())
	}
}

// Test Stop with a nil process (already stopped).
func TestManager_Stop_NilProcess(t *testing.T) {
	manager := NewManager(nil)

	// Should not panic
	_, err := manager.StopByPort(99999)
	if err == nil {
		t.Fatal("expected error for nonexistent port")
	}
}

// Test that the singleton GetManager returns the same instance.
func TestGetManager(t *testing.T) {
	m1 := GetManager()
	m2 := GetManager()
	if m1 != m2 {
		t.Fatal("GetManager should return the same singleton instance")
	}
}

// Test MergePassthroughFlags with empty passthrough.
func TestMergePassthroughFlags_EmptyPassthrough(t *testing.T) {
	computed := []string{"--model", "test.gguf", "--port", "8080"}
	merged := MergePassthroughFlags(computed, []string{})
	if len(merged) != len(computed) {
		t.Errorf("expected same number of flags with empty passthrough, got %d vs %d", len(merged), len(computed))
	}
}

// Test MergePassthroughFlags with boolean flag passthrough (no value).
func TestMergePassthroughFlags_BooleanFlag(t *testing.T) {
	computed := []string{"--model", "test.gguf", "--ctx-size", "4096"}
	passthrough := []string{"--jinja"}
	merged := MergePassthroughFlags(computed, passthrough)

	foundJinja := false
	for _, f := range merged {
		if f == "--jinja" {
			foundJinja = true
		}
	}
	if !foundJinja {
		t.Error("expected --jinja in merged flags")
	}

	// --ctx-size 4096 should remain since passthrough doesn't override it
	foundCtx := false
	for i, f := range merged {
		if f == "--ctx-size" && i+1 < len(merged) && merged[i+1] == "4096" {
			foundCtx = true
		}
	}
	if !foundCtx {
		t.Error("expected --ctx-size 4096 to remain in merged flags")
	}
}

// Test Stop with port=0 and empty model name returns error.
func TestManager_Stop_NoArgs(t *testing.T) {
	manager := NewManager(nil)

	_, err := manager.Stop(0, "")
	if err == nil {
		t.Fatal("expected error when calling Stop with no port or model name")
	}
}

// Test that Start with a mock command that outputs to stdout writes to the log file.
func TestManager_LogOutput(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	// Create a mock script that writes to stdout
	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\necho 'server started'\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19958, mockScript)
	_, err = manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	// Wait for output to be flushed
	time.Sleep(300 * time.Millisecond)

	logPath := filepath.Join(tmpDir, "llama-server-19958.log")
	content, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read log file: %v", err)
	}

	if len(content) == 0 {
		t.Error("expected log file to have content")
	}

	manager.StopByPort(19958)
}

// Test ProcessInfo.Status returns ProcessStopped for a process with nil cmd.
func TestProcessInfo_StatusNilCmd(t *testing.T) {
	proc := &ProcessInfo{
		PID: 9999,
		cmd: nil,
	}

	if proc.Status() != ProcessStopped {
		t.Errorf("expected stopped status for nil cmd, got %s", proc.Status())
	}
}

// Test StopByModelName with a partial name match via HasSuffix.
func TestManager_StopByModelName_PartialMatch(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19957, mockScript)
	proc, err := manager.Start(result, "models/llama-3.1-8b.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	// Verify it's running
	if proc.Status() != ProcessRunning {
		t.Errorf("expected running status, got %s", proc.Status())
	}

	// Should find it via the HasSuffix match against the model path
	stopped, err := manager.StopByModelName("llama-3.1-8b.gguf")
	if err != nil {
		t.Fatalf("StopByModelName() error: %v", err)
	}

	if len(stopped) == 0 {
		t.Fatal("expected to find process by partial model name")
	}

	// Give time for cleanup
	time.Sleep(200 * time.Millisecond)
	if stopped[0].Status() != ProcessStopped {
		t.Errorf("expected stopped status, got %s", stopped[0].Status())
	}
}

// Test that List returns processes sorted by start time.
func TestManager_List_SortedByStartTime(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	// Start multiple processes with small delays
	for i := 0; i < 3; i++ {
		result := makeTestResult(t, 19950+i, mockScript)
		_, err := manager.Start(result, fmt.Sprintf("model-%d.gguf", i), nil)
		if err != nil {
			t.Fatalf("Start() error for port %d: %v", 19950+i, err)
		}
		time.Sleep(10 * time.Millisecond)
	}

	procs := manager.List()
	if len(procs) != 3 {
		t.Fatalf("expected 3 processes, got %d", len(procs))
	}

	// Verify they are sorted by start time (earliest first)
	for i := 1; i < len(procs); i++ {
		if procs[i].StartTime.Before(procs[i-1].StartTime) {
			t.Errorf("processes not sorted by start time: process %d started before process %d", i, i-1)
		}
	}

	// Cleanup
	for i := 0; i < 3; i++ {
		manager.StopByPort(19950 + i)
	}
}

// Test that the Manager tracks processes in both procs and portMap maps.
func TestManager_TracksInBothMaps(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	port := 19949
	result := makeTestResult(t, port, mockScript)
	proc, err := manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}

	// Should be findable by PID
	if manager.GetByPID(proc.PID) == nil {
		t.Error("process not found in procs map by PID")
	}

	// Should be findable by port
	if manager.GetByPort(port) == nil {
		t.Error("process not found in portMap by port")
	}

	manager.StopByPort(port)

	// After stop, should be in neither map
	if manager.GetByPID(proc.PID) != nil {
		t.Error("process still in procs map after stop")
	}
	if manager.GetByPort(port) != nil {
		t.Error("process still in portMap after stop")
	}
}

func TestStartOptsStart_CPU_Device_Flags(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	oldWaitForReady := waitForReadyFunc
	t.Cleanup(func() {
		waitForReadyFunc = oldWaitForReady
	})
	waitForReadyFunc = func(*Manager, int, int, time.Duration) error { return nil }

	mockScript := filepath.Join(tmpDir, "mock-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\nwhile true; do sleep 1; done\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	port, err := manager.StartOptsStart(StartOpts{
		ModelPath:   "/tmp/test-model.gguf",
		LlamaServer: mockScript,
		ForceCPU:    true,
	})
	if err != nil {
		t.Fatalf("StartOptsStart() error: %v", err)
	}
	if port <= 0 {
		t.Fatalf("expected positive port, got %d", port)
	}

	proc := manager.GetByPort(port)
	if proc == nil {
		t.Fatal("process not found by port")
	}

	// Verify --n-gpu-layers 0 is in flags
	hasNPGLayers0 := false
	for i, f := range proc.Flags {
		if f == "--n-gpu-layers" && i+1 < len(proc.Flags) && proc.Flags[i+1] == "0" {
			hasNPGLayers0 = true
			break
		}
	}
	if !hasNPGLayers0 {
		t.Errorf("expected --n-gpu-layers 0 in flags, got: %v", proc.Flags)
	}

	// Verify both GPU selectors are empty (GPU hidden in CPU mode)
	var cudaVisibleDevices string
	var hipVisibleDevices string
	var vkVisibleDevices string
	for _, env := range proc.cmd.Env {
		if strings.HasPrefix(env, "CUDA_VISIBLE_DEVICES=") {
			cudaVisibleDevices = env
		}
		if strings.HasPrefix(env, "HIP_VISIBLE_DEVICES=") {
			hipVisibleDevices = env
		}
		if strings.HasPrefix(env, "GGML_VK_VISIBLE_DEVICES=") {
			vkVisibleDevices = env
		}
	}
	if cudaVisibleDevices != "CUDA_VISIBLE_DEVICES=" {
		t.Errorf("expected CUDA_VISIBLE_DEVICES=\"\" in environment, got: %s", cudaVisibleDevices)
	}
	if hipVisibleDevices != "HIP_VISIBLE_DEVICES=" {
		t.Errorf("expected HIP_VISIBLE_DEVICES=\"\" in environment, got: %s", hipVisibleDevices)
	}
	if vkVisibleDevices != "GGML_VK_VISIBLE_DEVICES=" {
		t.Errorf("expected GGML_VK_VISIBLE_DEVICES=\"\" in environment, got: %s", vkVisibleDevices)
	}

	// Verify GPUIndex is "cpu"
	if proc.GPUIndex != "cpu" {
		t.Errorf("expected GPUIndex \"cpu\", got %q", proc.GPUIndex)
	}

	manager.StopByPort(port)
}

func TestBuildChildEnvGPUIsolation(t *testing.T) {
	t.Setenv("CUDA_VISIBLE_DEVICES", "stale")

	env := buildChildEnv(runtimemgr.BuildBackendCUDA, 1, false, "/opt/anvil/runtimes/r1/llama-server", nil)
	want := "CUDA_VISIBLE_DEVICES=1"
	found := false
	stale := false
	for _, e := range env {
		if e == want {
			found = true
		}
		if e == "CUDA_VISIBLE_DEVICES=stale" {
			stale = true
		}
	}
	if !found {
		t.Errorf("expected env to contain %q, got %v", want, envContainingCUDA(env))
	}
	if stale {
		t.Errorf("stale CUDA_VISIBLE_DEVICES leaked into child env")
	}
}

func TestBuildChildEnvCPUFallback(t *testing.T) {
	env := buildChildEnv(runtimemgr.BuildBackendCUDA, -1, true, "", nil)
	wantCUDA := "CUDA_VISIBLE_DEVICES="
	wantHIP := "HIP_VISIBLE_DEVICES="
	wantVK := "GGML_VK_VISIBLE_DEVICES="
	foundCUDA := false
	foundHIP := false
	foundVK := false
	for _, e := range env {
		if e == wantCUDA {
			foundCUDA = true
		}
		if e == wantHIP {
			foundHIP = true
		}
		if e == wantVK {
			foundVK = true
		}
	}
	if !foundCUDA || !foundHIP || !foundVK {
		t.Errorf("expected empty GPU selectors, got %v", env)
	}
}

func TestBuildChildEnvNoGPUIndexLeavesUnset(t *testing.T) {
	// No GPU and not forced CPU: don't override CUDA_VISIBLE_DEVICES at all.
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	env := buildChildEnv(runtimemgr.BuildBackendCUDA, -1, false, "", nil)
	for _, e := range env {
		if strings.HasPrefix(e, "CUDA_VISIBLE_DEVICES=") {
			t.Errorf("expected no CUDA_VISIBLE_DEVICES entry, got %s", e)
		}
		if strings.HasPrefix(e, "GGML_VK_VISIBLE_DEVICES=") {
			t.Errorf("expected no GGML_VK_VISIBLE_DEVICES entry, got %s", e)
		}
	}
}

func TestBuildChildEnvVulkanUsesGGMLVKVisibleDevices(t *testing.T) {
	t.Setenv("GGML_VK_VISIBLE_DEVICES", "stale")
	env := buildChildEnv(runtimemgr.BuildBackendVulkan, 2, false, "/opt/runtimes/llama-vulkan/llama-server", nil)
	want := "GGML_VK_VISIBLE_DEVICES=2"
	found := false
	for _, e := range env {
		if e == want {
			found = true
		}
		if strings.HasPrefix(e, "CUDA_VISIBLE_DEVICES=") {
			t.Fatalf("unexpected CUDA selector in Vulkan env: %v", envContainingCUDA(env))
		}
	}
	if !found {
		t.Fatalf("expected env to contain %q, got %v", want, env)
	}
}

func TestBuildChildEnvROCmUsesHIPVisibleDevices(t *testing.T) {
	t.Setenv("HIP_VISIBLE_DEVICES", "stale")
	env := buildChildEnv(runtimemgr.BuildBackendROCm, 3, false, "/opt/runtimes/llama-rocm/llama-server", nil)
	want := "HIP_VISIBLE_DEVICES=3"
	found := false
	for _, e := range env {
		if e == want {
			found = true
		}
		if strings.HasPrefix(e, "CUDA_VISIBLE_DEVICES=") {
			t.Fatalf("unexpected CUDA selector in ROCm env: %v", envContainingCUDA(env))
		}
	}
	if !found {
		t.Fatalf("expected env to contain %q, got %v", want, env)
	}
}

func TestParseCUDADeviceIndex(t *testing.T) {
	cases := map[string]int{
		"cuda:0": 0,
		"cuda:1": 1,
		"cuda:7": 7,
		"cpu":    -1,
		"":       -1,
		"cuda:":  -1,
	}
	for in, want := range cases {
		if got := parseCUDADeviceIndex(in); got != want {
			t.Errorf("parseCUDADeviceIndex(%q) = %d, want %d", in, got, want)
		}
	}
}

func envContainingCUDA(env []string) []string {
	out := make([]string, 0)
	for _, e := range env {
		if strings.HasPrefix(e, "CUDA_VISIBLE_DEVICES=") {
			out = append(out, e)
		}
	}
	return out
}

func TestBuildChildEnvSetsLDLibraryPath(t *testing.T) {
	t.Setenv("LD_LIBRARY_PATH", "")
	env := buildChildEnv(runtimemgr.BuildBackendCUDA, 0, false, "/opt/runtimes/llama-b9275/llama-server", nil)
	wantPrefix := "LD_LIBRARY_PATH=/opt/runtimes/llama-b9275"
	for _, e := range env {
		if e == wantPrefix {
			return
		}
	}
	t.Errorf("expected env to contain %q, got %v", wantPrefix, ldEntries(env))
}

func TestBuildChildEnvPrependsToExistingLDP(t *testing.T) {
	t.Setenv("LD_LIBRARY_PATH", "/usr/local/lib:/opt/foo/lib")
	env := buildChildEnv(runtimemgr.BuildBackendCUDA, 0, false, "/opt/runtimes/llama-b9275/llama-server", nil)
	want := "LD_LIBRARY_PATH=" + strings.Join([]string{
		"/opt/runtimes/llama-b9275",
		strings.Join([]string{"/usr/local/lib", "/opt/foo/lib"}, string(os.PathListSeparator)),
	}, string(os.PathListSeparator))
	for _, e := range env {
		if e == want {
			return
		}
	}
	t.Errorf("expected env to contain %q, got %v", want, ldEntries(env))
}

func TestBuildChildEnvPreservesConfiguredLDP(t *testing.T) {
	t.Setenv("LD_LIBRARY_PATH", "/parent/lib")
	env := buildChildEnv(runtimemgr.BuildBackendCUDA, 0, false, "/opt/runtimes/llama-b9275/llama-server", map[string]string{
		"LD_LIBRARY_PATH": strings.Join([]string{"/config/cuda/lib", "/config/blas/lib"}, string(os.PathListSeparator)),
	})
	want := "LD_LIBRARY_PATH=" + strings.Join([]string{
		"/opt/runtimes/llama-b9275",
		strings.Join([]string{"/config/cuda/lib", "/config/blas/lib"}, string(os.PathListSeparator)),
		"/parent/lib",
	}, string(os.PathListSeparator))
	for _, e := range env {
		if e == want {
			return
		}
	}
	t.Errorf("expected env to contain %q, got %v", want, ldEntries(env))
}

func TestBuildChildEnvEmptyBinaryPreservesExistingLDP(t *testing.T) {
	t.Setenv("LD_LIBRARY_PATH", "/keep/this")
	env := buildChildEnv(runtimemgr.BuildBackendCUDA, 0, false, "", nil)
	want := "LD_LIBRARY_PATH=/keep/this"
	for _, e := range env {
		if e == want {
			return
		}
	}
	t.Errorf("expected parent LD_LIBRARY_PATH preserved, got %v", ldEntries(env))
}

func TestBuildChildEnvAddsExtraEnv(t *testing.T) {
	env := buildChildEnv(runtimemgr.BuildBackendCUDA, 0, false, "/opt/runtimes/llama-b9275/llama-server", map[string]string{
		"GGML_VK_VISIBLE_DEVICES": "1",
	})
	want := "GGML_VK_VISIBLE_DEVICES=1"
	for _, e := range env {
		if e == want {
			return
		}
	}
	t.Errorf("expected env to contain %q, got %v", want, env)
}

func TestDiagnoseCrashPatterns(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	cases := []struct {
		name       string
		logContent string
		wantSubstr string
	}{
		{
			name:       "bar1",
			logContent: "NVRM: BAR1 is 0M\n",
			wantSubstr: "Above 4G Decoding",
		},
		{
			name:       "arch",
			logContent: "ptxas fatal : Unsupported gpu architecture 'sm_120'\n",
			wantSubstr: "different GPU architecture",
		},
		{
			name:       "oom",
			logContent: "CUDA error: out of memory\n",
			wantSubstr: "Not enough VRAM",
		},
		{
			name:       "unsupported-model",
			logContent: "unknown model architecture: qwen3.6\n",
			wantSubstr: "not supported by your llama-server version",
		},
		{
			name:       "generic",
			logContent: "some unrelated crash text\n",
			wantSubstr: "runtime is too old for this model",
		},
	}

	for i, tc := range cases {
		logPath := filepath.Join(tmpDir, "llama-server-12345.log")
		if err := os.WriteFile(logPath, []byte(tc.logContent), 0o644); err != nil {
			t.Fatal(err)
		}
		err := manager.diagnoseCrash(12345)
		if err == nil {
			t.Fatalf("%s: expected error", tc.name)
		}
		if !strings.Contains(err.Error(), tc.wantSubstr) {
			t.Fatalf("%s[%d]: error %q does not contain %q", tc.name, i, err.Error(), tc.wantSubstr)
		}
	}
}

func TestWaitForReadyDetectsImmediateCrash(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "crash-llama-server")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\necho 'ptxas fatal : Unsupported gpu architecture' 1>&2\nexit 1\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19945, mockScript)
	proc, err := manager.Start(result, "test-model.gguf", nil)
	if err != nil {
		t.Fatalf("Start() error: %v", err)
	}
	<-proc.done

	err = manager.waitForReady(proc.Port, proc.PID, 200*time.Millisecond)
	if err == nil {
		t.Fatal("expected readiness check to fail for crashed process")
	}
	if !strings.Contains(err.Error(), "different GPU architecture") {
		t.Fatalf("waitForReady error = %v, want GPU architecture guidance", err)
	}
}

func TestManager_Start_ReportsSharedLibraryCrash(t *testing.T) {
	tmpDir := t.TempDir()
	manager := NewManager(nil)
	manager.SetLogDir(tmpDir)

	mockScript := filepath.Join(tmpDir, "shared-lib-crash")
	err := os.WriteFile(mockScript, []byte("#!/bin/sh\necho 'error while loading shared libraries: libllama-common.so.0: file too short' 1>&2\nexit 1\n"), 0o755)
	if err != nil {
		t.Fatalf("failed to create mock script: %v", err)
	}

	result := makeTestResult(t, 19944, mockScript)
	result.ReadyTimeout = 2 * time.Second

	_, err = manager.Start(result, "test-model.gguf", nil)
	if err == nil {
		t.Fatal("expected Start() to fail for shared library crash")
	}
	if strings.Contains(err.Error(), "timed out after") {
		t.Fatalf("expected crash detection, got timeout: %v", err)
	}
	if !strings.Contains(err.Error(), "file too short") {
		t.Fatalf("Start() error = %v, want shared library stderr", err)
	}
}

func ldEntries(env []string) []string {
	out := make([]string, 0)
	for _, e := range env {
		if strings.HasPrefix(e, "LD_LIBRARY_PATH=") {
			out = append(out, e)
		}
	}
	return out
}
