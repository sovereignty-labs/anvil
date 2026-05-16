package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/hirdforge/nollama/internal/config"
	"github.com/hirdforge/nollama/internal/hardware"
	"github.com/hirdforge/nollama/internal/process"
	"github.com/spf13/cobra"
)

type daemonStatusResponse struct {
	Models []daemonStatusModel `json:"models"`
	Node   daemonStatusNode    `json:"node"`
}

type daemonStatusModel struct {
	Name          string `json:"name"`
	Port          int    `json:"port"`
	GPU           string `json:"gpu"`
	PID           int    `json:"pid"`
	UptimeSeconds int64  `json:"uptime_seconds"`
}

type daemonStatusNode struct {
	GPUs       []daemonStatusGPU `json:"gpus"`
	RAMTotalMB uint64            `json:"ram_total_mb"`
	RAMFreeMB  uint64            `json:"ram_free_mb"`
}

type daemonStatusGPU struct {
	Index       int    `json:"index"`
	Name        string `json:"name"`
	VRAMTotalMB uint64 `json:"vram_total_mb"`
	VRAMFreeMB  uint64 `json:"vram_free_mb"`
}

type statusReport struct {
	Models []statusModel
	Node   statusNode
}

type statusModel struct {
	Name   string
	Port   int
	GPU    string
	PID    int
	Uptime time.Duration
}

type statusNode struct {
	GPUs       []statusGPU
	RAMTotalMB uint64
	RAMFreeMB  uint64
}

type statusGPU struct {
	Index       int
	Name        string
	VRAMTotalMB uint64
	VRAMFreeMB  uint64
}

func runStatus(_ *cobra.Command, _ []string) error {
	daemonAddr := resolveStatusDaemonAddr()

	report, daemonUnavailable, err := fetchDaemonStatus(daemonAddr)
	if err != nil && !daemonUnavailable {
		return err
	}
	if daemonUnavailable {
		local := buildLocalStatusReport()
		if len(local.Models) == 0 {
			fmt.Printf("No nollama daemon detected on %s\n", daemonAddr)
			fmt.Println("No loaded models. Run `nollama serve --config` or `nollama load <model.gguf>`")
			return nil
		}
		renderStatusReport(local)
		return nil
	}

	renderStatusReport(*report)
	return nil
}

func resolveStatusDaemonAddr() string {
	addr := "localhost:11434"

	if cfgPath := config.FindConfig(); cfgPath != "" {
		if cfg, err := config.Load(cfgPath); err == nil && cfg.Bind != "" {
			addr = normalizeDialAddress(cfg.Bind)
		}
	}

	return addr
}

func normalizeDialAddress(bind string) string {
	bind = strings.TrimSpace(bind)
	if bind == "" {
		return "localhost:11434"
	}

	if strings.Contains(bind, "://") {
		if u, err := url.Parse(bind); err == nil && u.Host != "" {
			return normalizeDialAddress(u.Host)
		}
	}

	host, port, err := net.SplitHostPort(bind)
	if err != nil {
		return bind
	}
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "localhost"
	}
	return net.JoinHostPort(host, port)
}

func fetchDaemonStatus(addr string) (*statusReport, bool, error) {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://%s/api/status", addr))
	if err != nil {
		if isDaemonUnavailable(err) {
			return nil, true, err
		}
		return nil, false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, false, fmt.Errorf("daemon at %s returned %s", addr, resp.Status)
	}

	var d daemonStatusResponse
	if err := json.NewDecoder(resp.Body).Decode(&d); err != nil {
		return nil, false, fmt.Errorf("decoding daemon status: %w", err)
	}

	return convertDaemonStatus(d), false, nil
}

func isDaemonUnavailable(err error) bool {
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	return errors.Is(err, context.DeadlineExceeded)
}

func convertDaemonStatus(d daemonStatusResponse) *statusReport {
	report := &statusReport{
		Models: make([]statusModel, 0, len(d.Models)),
		Node: statusNode{
			GPUs:       make([]statusGPU, 0, len(d.Node.GPUs)),
			RAMTotalMB: d.Node.RAMTotalMB,
			RAMFreeMB:  d.Node.RAMFreeMB,
		},
	}

	for _, gpu := range d.Node.GPUs {
		report.Node.GPUs = append(report.Node.GPUs, statusGPU{
			Index:       gpu.Index,
			Name:        gpu.Name,
			VRAMTotalMB: gpu.VRAMTotalMB,
			VRAMFreeMB:  gpu.VRAMFreeMB,
		})
	}
	for _, model := range d.Models {
		report.Models = append(report.Models, statusModel{
			Name:   model.Name,
			Port:   model.Port,
			GPU:    model.GPU,
			PID:    model.PID,
			Uptime: time.Duration(model.UptimeSeconds) * time.Second,
		})
	}

	return report
}

func buildLocalStatusReport() statusReport {
	report := statusReport{
		Models: make([]statusModel, 0),
	}

	hw, err := hardware.Detect()
	if err == nil && hw != nil {
		report.Node = statusNode{
			GPUs:       make([]statusGPU, 0, len(hw.GPUs)),
			RAMTotalMB: hw.CPU.RAMTotalMB,
			RAMFreeMB:  hw.CPU.RAMFreeMB,
		}
		for _, gpu := range hw.GPUs {
			report.Node.GPUs = append(report.Node.GPUs, statusGPU{
				Index:       gpu.Index,
				Name:        gpu.DisplayName(),
				VRAMTotalMB: gpu.VRAMTotal,
				VRAMFreeMB:  gpu.VRAMFree,
			})
		}
	}

	procs := process.GetManager().List()
	for _, proc := range procs {
		report.Models = append(report.Models, statusModel{
			Name:   proc.ModelName,
			Port:   proc.Port,
			GPU:    proc.GPUIndex,
			PID:    proc.PID,
			Uptime: proc.Uptime(),
		})
	}

	return report
}

func renderStatusReport(report statusReport) {
	rows := buildStatusRows(report)
	if len(rows) == 0 {
		fmt.Println("No loaded models.")
		fmt.Println()
		fmt.Println("MODELS: 0 loaded")
		return
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "NODE\tGPU\tMODEL\tPORT\tPID\tUPTIME")
	for _, row := range rows {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n",
			row.Node, row.GPU, row.Model, row.Port, row.PID, row.Uptime)
	}
	w.Flush()

	fmt.Println()
	fmt.Printf("MODELS: %d loaded\n", len(report.Models))
}

type statusRow struct {
	Node   string
	GPU    string
	Model  string
	Port   string
	PID    string
	Uptime string
}

func buildStatusRows(report statusReport) []statusRow {
	rows := make([]statusRow, 0, len(report.Models)+len(report.Node.GPUs))
	usedGPU := make(map[int]bool)
	gpuNames := make(map[int]string, len(report.Node.GPUs))
	for _, gpu := range report.Node.GPUs {
		gpuNames[gpu.Index] = gpu.Name
	}

	for _, model := range report.Models {
		gpuLabel := formatModelGPU(model.GPU, gpuNames)
		if idx, ok := parseCUDAIndex(model.GPU); ok {
			usedGPU[idx] = true
		}
		rows = append(rows, statusRow{
			Node:   "local",
			GPU:    gpuLabel,
			Model:  model.Name,
			Port:   fmt.Sprintf("%d", model.Port),
			PID:    fmt.Sprintf("%d", model.PID),
			Uptime: process.FormatDuration(model.Uptime),
		})
	}

	for _, gpu := range report.Node.GPUs {
		if usedGPU[gpu.Index] {
			continue
		}
		rows = append(rows, statusRow{
			Node:   "local",
			GPU:    fmt.Sprintf("cuda:%d (%s)", gpu.Index, gpu.Name),
			Model:  "(idle)",
			Port:   "—",
			PID:    "—",
			Uptime: "—",
		})
	}

	return rows
}

func formatModelGPU(gpu string, names map[int]string) string {
	if idx, ok := parseCUDAIndex(gpu); ok {
		if name, ok := names[idx]; ok && name != "" {
			return fmt.Sprintf("%s (%s)", gpu, name)
		}
	}
	return gpu
}

func parseCUDAIndex(gpu string) (int, bool) {
	if !strings.HasPrefix(gpu, "cuda:") {
		return 0, false
	}
	var idx int
	if _, err := fmt.Sscanf(gpu, "cuda:%d", &idx); err != nil {
		return 0, false
	}
	return idx, true
}
