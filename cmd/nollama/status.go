package main

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"github.com/hirdforge/nollama/internal/config"
	"github.com/hirdforge/nollama/internal/federation"
	"github.com/spf13/cobra"
)

type statusRow struct {
	Node   string
	GPU    string
	Model  string
	VRAM   string
	Port   string
	PID    string
	Uptime string
}

type nodeStatusResult struct {
	Name   string
	Status *federation.StatusResponse
	Err    error
}

func runStatus(cmd *cobra.Command, _ []string) error {
	if client, err := resolveNodeClient(cmd); err != nil {
		return err
	} else if client != nil {
		nodeName, _ := cmd.Flags().GetString("node")
		resp, err := statusForClient(client, 5*time.Second)
		if err != nil {
			return err
		}
		renderSingleNodeStatus(nodeName, resp)
		return nil
	}

	registry, cfgRemotes, err := loadFederatedRemotes()
	if err != nil {
		return err
	}

	localURL := resolveStatusDaemonAddr()
	mergedRemotes := federation.MergeRemotes(registry, cfgRemotes)
	targets := []nodeTarget{{Name: "local", URL: localURL}}
	remoteNames := sortedRemoteNames(mergedRemotes)
	for _, name := range remoteNames {
		targets = append(targets, nodeTarget{Name: name, URL: mergedRemotes[name]})
	}

	results := queryNodeStatuses(targets, 5*time.Second)

	if len(remoteNames) == 0 && len(results) == 1 && results[0].Err != nil {
		fmt.Println("No daemon running and no remotes configured.")
		return nil
	}

	renderFleetStatus(results)
	return nil
}

type nodeTarget struct {
	Name string
	URL  string
}

func queryNodeStatuses(targets []nodeTarget, timeout time.Duration) []nodeStatusResult {
	results := make(chan nodeStatusResult, len(targets))
	var wg sync.WaitGroup

	for _, target := range targets {
		target := target
		wg.Add(1)
		go func() {
			defer wg.Done()

			client := federation.NewClient(target.URL)
			client.HTTPClient.Timeout = timeout

			status, err := client.Status()
			results <- nodeStatusResult{Name: target.Name, Status: status, Err: err}
		}()
	}

	wg.Wait()
	close(results)

	out := make([]nodeStatusResult, 0, len(targets))
	for result := range results {
		out = append(out, result)
	}

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Name == out[j].Name {
			return out[i].Err == nil && out[j].Err != nil
		}
		return out[i].Name < out[j].Name
	})

	return out
}

func statusForClient(client *federation.Client, timeout time.Duration) (*federation.StatusResponse, error) {
	client.HTTPClient.Timeout = timeout
	return client.Status()
}

func renderSingleNodeStatus(nodeName string, resp *federation.StatusResponse) {
	rows := buildStatusRows(nodeName, resp)
	renderStatusRows(rows)
}

func renderFleetStatus(results []nodeStatusResult) {
	rows := make([]statusRow, 0)
	online := 0

	for _, result := range results {
		if result.Err != nil || result.Status == nil {
			rows = append(rows, statusRow{
				Node:   result.Name,
				GPU:    "offline",
				Model:  "—",
				VRAM:   "—",
				Port:   "—",
				PID:    "—",
				Uptime: "—",
			})
			continue
		}

		online++
		rows = append(rows, buildStatusRows(result.Name, result.Status)...)
	}

	if len(rows) == 0 {
		fmt.Println("No loaded models.")
		return
	}

	renderStatusRows(rows)
	fmt.Println()
	fmt.Printf("NODES: %d online\n", online)
}

func renderStatusRows(rows []statusRow) {
	if len(rows) == 0 {
		fmt.Println("No loaded models.")
		return
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "NODE\tGPU\tMODEL\tVRAM\tPORT\tPID\tUPTIME")
	for _, row := range rows {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			row.Node, row.GPU, row.Model, row.VRAM, row.Port, row.PID, row.Uptime)
	}
	_ = w.Flush()
}

func buildStatusRows(nodeName string, resp *federation.StatusResponse) []statusRow {
	rows := make([]statusRow, 0, len(resp.Models))
	for _, model := range resp.Models {
		rows = append(rows, statusRow{
			Node:   nodeName,
			GPU:    modelGPUName(resp.Node, model.GPU),
			Model:  model.Name,
			VRAM:   modelVRAM(resp.Node, model.GPU),
			Port:   fmt.Sprintf("%d", model.Port),
			PID:    fmt.Sprintf("%d", model.PID),
			Uptime: processFormatDuration(time.Duration(model.UptimeSeconds) * time.Second),
		})
	}

	return rows
}

func modelGPUName(node federation.StatusNode, gpu string) string {
	if gpu == "" || gpu == "cpu" {
		return "cpu"
	}
	if idx, ok := parseCUDAIndex(gpu); ok {
		for _, g := range node.GPUs {
			if g.Index == idx && g.Name != "" {
				return g.Name
			}
		}
	}
	return gpu
}

func modelVRAM(node federation.StatusNode, gpu string) string {
	if gpu == "" || gpu == "cpu" {
		return "—"
	}
	if idx, ok := parseCUDAIndex(gpu); ok {
		for _, g := range node.GPUs {
			if g.Index == idx {
				return fmt.Sprintf("%.1f/%.1fGB", mbToGB(g.VRAMFreeMB), mbToGB(g.VRAMTotalMB))
			}
		}
	}
	return "—"
}

func mbToGB(mb uint64) float64 {
	return float64(mb) / 1024
}

func processFormatDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	days := d / (24 * time.Hour)
	d -= days * 24 * time.Hour
	hours := d / time.Hour
	if days > 0 {
		return fmt.Sprintf("%dd %dh", days, hours)
	}
	if hours > 0 {
		return fmt.Sprintf("%dh %dm", hours, (d%time.Hour)/time.Minute)
	}
	minutes := d / time.Minute
	if minutes > 0 {
		return fmt.Sprintf("%dm", minutes)
	}
	seconds := d / time.Second
	return fmt.Sprintf("%ds", seconds)
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
