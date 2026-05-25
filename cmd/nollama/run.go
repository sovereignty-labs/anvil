package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/sovereignty-labs/nollama/internal/config"
	"github.com/sovereignty-labs/nollama/internal/hardware"
	"github.com/sovereignty-labs/nollama/internal/model"
	"github.com/sovereignty-labs/nollama/internal/process"
	"github.com/spf13/cobra"
)

var runModelCmd = &cobra.Command{
	Use:   "run <model>",
	Short: "Load a model and start an interactive chat",
	Long: `Load a model and start an interactive chat session.

The model can be a bare filename, a path, or a name without .gguf extension.
If nollama serve is running and the model is already loaded, run connects
through the proxy instead of spawning a second instance.

Examples:
  nollama run google_gemma-4-E2B-it-Q4_K_M
  nollama run ~/models/my-model.gguf
  nollama run Qwen3-4B-Q4_K_M --gpu 0`,
	Args: exactPositionalArgs(1),
	RunE: runRunModel,
}

func init() {
	runModelCmd.Flags().Int("gpu", -1, "GPU index to load on")
	runModelCmd.Flags().Bool("cpu", false, "Force CPU inference")
	runModelCmd.Flags().String("runtime", "", "Use a specific llama-server runtime")
	runModelCmd.Flags().Int("port", 0, "Pin llama-server to this port instead of auto-assigning")
	runModelCmd.Flags().StringArray("passthrough", nil, "Extra llama-server flags (repeatable)")
	runModelCmd.Flags().Duration("ready-timeout", 120*time.Second, "How long to wait for llama-server to become healthy")
}

func runRunModel(cmd *cobra.Command, args []string) error {
	modelPath, err := resolveModelPath(args[0])
	if err != nil {
		return err
	}
	modelName := filepath.Base(modelPath)

	cfg, err := loadCLIConfig()
	if err != nil {
		return err
	}

	// If nollama serve is running and already hosts this model, route the chat
	// through its proxy and skip spawning a second instance.
	if endpoint, ok := findDaemonEndpointForModel(cfg, modelName); ok {
		fmt.Fprintf(os.Stderr, "Connecting to running daemon: %s\n", endpoint)
		return runChatAgainst(endpoint, modelStem(modelName), false)
	}

	runtimeName, _ := cmd.Flags().GetString("runtime")
	endpoint, cleanup, err := startLlamaServerForRun(cmd, args, cfg, modelPath, modelName, runtimeName)
	if err != nil {
		return err
	}
	defer cleanup()

	return runChatAgainst(endpoint, modelStem(modelName), true)
}

// startLlamaServerForRun spawns a llama-server, waits for it to become healthy,
// and returns the OpenAI-compatible base URL plus a cleanup func that kills
// the spawned process.
func startLlamaServerForRun(cmd *cobra.Command, args []string, cfg *config.Config, modelPath, modelName, runtimeName string) (string, func(), error) {
	llamaServerFlag, err := resolveLlamaServerPathWithRuntime(cmd, cfg, runtimeName)
	if err != nil {
		return "", func() {}, err
	}

	fmt.Fprintln(os.Stderr, "Parsing GGUF metadata...")
	meta, err := model.ParseGGUF(modelPath)
	if err != nil {
		return "", func() {}, fmt.Errorf("failed to parse GGUF: %w", err)
	}

	fmt.Fprintln(os.Stderr, "Detecting hardware...")
	inv, err := hardware.Detect()
	if err != nil {
		return "", func() {}, fmt.Errorf("hardware detection failed: %w", err)
	}

	result, err := process.ComputeFlags(meta, modelPath, inv, llamaServerFlag, 0)
	if err != nil {
		return "", func() {}, fmt.Errorf("flag computation failed: %w", err)
	}
	if pinnedPort, _ := cmd.Flags().GetInt("port"); pinnedPort > 0 {
		process.OverrideResultPort(result, pinnedPort)
	}
	if forceCPU, _ := cmd.Flags().GetBool("cpu"); forceCPU {
		// ComputeFlags already handles CPU fallback when no GPU fits; this is
		// the explicit-opt-in path. We just mark the result so buildChildEnv
		// hides the GPUs.
		result.CPUFallback = true
		result.SelectedDevice = "cpu"
	} else if gpu, _ := cmd.Flags().GetInt("gpu"); gpu >= 0 {
		result.SelectedDevice = fmt.Sprintf("cuda:%d", gpu)
		result.GPUIndex = gpu
	}
	result.Command = buildCommand(llamaServerFlag, result.Flags)

	passthrough := collectPassthrough(cmd, args)

	manager := process.GetManager()
	for _, proc := range manager.List() {
		if proc.ModelName == modelName || proc.ModelPath == modelPath {
			endpoint := fmt.Sprintf("http://127.0.0.1:%d", proc.Port)
			fmt.Fprintf(os.Stderr, "Reusing already-running llama-server (PID %d, port %d)\n", proc.PID, proc.Port)
			return endpoint, func() {}, nil
		}
	}

	fmt.Fprintln(os.Stderr, "Starting llama-server...")
	procInfo, err := manager.Start(result, modelName, passthrough)
	if err != nil {
		return "", func() {}, fmt.Errorf("failed to start llama-server: %w", err)
	}
	procInfo.ModelPath = modelPath

	endpoint := fmt.Sprintf("http://127.0.0.1:%d", procInfo.Port)
	timeout, _ := cmd.Flags().GetDuration("ready-timeout")
	if err := waitForReady(endpoint, timeout); err != nil {
		// Show the tail of the log so the user knows why startup failed.
		printLogTail(manager.LogDir(), procInfo.Port, 10)
		_, _ = manager.StopByPort(procInfo.Port)
		return "", func() {}, fmt.Errorf("llama-server did not become ready: %w", err)
	}

	cleanup := func() {
		fmt.Fprintln(os.Stderr, "\nStopping llama-server...")
		_, _ = manager.StopByPort(procInfo.Port)
	}
	return endpoint, cleanup, nil
}

// runChatAgainst opens the interactive chat loop against endpoint and wires
// up Ctrl+C handling. When ownedProcess is true, the first Ctrl+C cancels the
// in-flight stream rather than exiting; a second Ctrl+C exits. When false
// (daemon mode), Ctrl+C just exits.
func runChatAgainst(endpoint, modelName string, ownedProcess bool) error {
	interrupt := make(chan struct{}, 1)
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	done := make(chan error, 1)
	go func() {
		done <- chatLoop(endpoint, modelName, interrupt, os.Stdin, os.Stdout)
	}()

	for {
		select {
		case <-sigCh:
			select {
			case interrupt <- struct{}{}:
			default:
			}
			// A second SIGINT before the loop finishes hard-exits so the user
			// isn't stuck if the chat loop wedges.
			select {
			case <-time.After(100 * time.Millisecond):
			case <-sigCh:
				return errors.New("interrupted")
			case err := <-done:
				return err
			}
		case err := <-done:
			return err
		}
	}
}

// findDaemonEndpointForModel asks nollama serve (at cfg.Bind, or the default
// 127.0.0.1:11434) whether modelName is already loaded. Returns the daemon's
// base URL when so; otherwise (no daemon, daemon reachable but model absent)
// reports false.
func findDaemonEndpointForModel(cfg *config.Config, modelName string) (string, bool) {
	bind := strings.TrimSpace(cfg.Bind)
	if bind == "" {
		bind = "127.0.0.1:11434"
	}
	// Bind may be 0.0.0.0:PORT; replace with 127.0.0.1 for the client probe.
	host, port, err := splitHostPort(bind)
	if err != nil {
		return "", false
	}
	if host == "" || host == "0.0.0.0" {
		host = "127.0.0.1"
	}
	endpoint := fmt.Sprintf("http://%s:%s", host, port)

	client := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := client.Get(endpoint + "/api/status")
	if err != nil {
		return "", false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", false
	}

	var status struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return "", false
	}
	targetStem := strings.ToLower(strings.TrimSuffix(modelName, ".gguf"))
	for _, m := range status.Models {
		if strings.ToLower(strings.TrimSuffix(m.Name, ".gguf")) == targetStem {
			return endpoint, true
		}
	}
	return "", false
}

// waitForReady polls /health every 500ms until it returns 200, or until
// timeout elapses. Prints a single ticking spinner so the user knows
// nollama is alive while a big model loads into VRAM.
func waitForReady(endpoint string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 1 * time.Second}
	tickers := []string{"|", "/", "-", "\\"}
	tick := 0
	first := true

	for time.Now().Before(deadline) {
		resp, err := client.Get(endpoint + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				if !first {
					fmt.Fprint(os.Stderr, "\r          \r")
				}
				return nil
			}
		}
		first = false
		fmt.Fprintf(os.Stderr, "\rWaiting for llama-server %s", tickers[tick%len(tickers)])
		tick++
		time.Sleep(500 * time.Millisecond)
	}
	fmt.Fprint(os.Stderr, "\r          \r")
	return fmt.Errorf("timed out after %s — model may be too large for available VRAM", timeout)
}

// printLogTail prints the last n lines of the llama-server log file to stderr.
func printLogTail(logDir string, port, n int) {
	path := filepath.Join(logDir, fmt.Sprintf("llama-server-%d.log", port))
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not read log %s: %v\n", path, err)
		return
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	fmt.Fprintf(os.Stderr, "\n--- last %d lines of %s ---\n", len(lines), path)
	fmt.Fprintln(os.Stderr, strings.Join(lines, "\n"))
}

// modelStem strips a trailing .gguf so the model name sent in chat requests
// matches what llama-server / nollamas /v1/models advertises.
func modelStem(name string) string {
	return strings.TrimSuffix(name, ".gguf")
}

// splitHostPort splits "host:port" or ":port" without pulling in net.SplitHostPort's
// IPv6-bracket strictness. Returns ("", "", err) if no colon is found.
func splitHostPort(s string) (string, string, error) {
	idx := strings.LastIndex(s, ":")
	if idx < 0 {
		return "", "", fmt.Errorf("no port in %q", s)
	}
	return s[:idx], s[idx+1:], nil
}

// drainBody is here so import "io" stays referenced even when none of the
// public surfaces above use it directly (defensive — readers may evolve).
var _ = io.Discard
