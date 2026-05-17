package mcp

import (
	"context"
	"fmt"
	"net/http"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"github.com/hirdforge/nollama/internal/config"
	"github.com/hirdforge/nollama/internal/federation"
	"github.com/hirdforge/nollama/internal/hardware"
	"github.com/hirdforge/nollama/internal/model"
	"github.com/hirdforge/nollama/internal/runtime"
	mcpkit "github.com/mark3labs/mcp-go/mcp"
)

func (r *Runner) registerTools() {
	r.server.AddTool(mcpkit.NewTool("nollama_status",
		mcpkit.WithDescription("Fleet overview across local and remote nodes"),
		mcpkit.WithString("node",
			mcpkit.Description("Filter to a specific remote node"),
		),
	), r.toolStatus)
	r.server.AddTool(mcpkit.NewTool("nollama_load",
		mcpkit.WithDescription("Load a model on a local or remote node"),
		mcpkit.WithString("model",
			mcpkit.Required(),
			mcpkit.Description("GGUF filename or fuzzy match"),
		),
		mcpkit.WithString("node",
			mcpkit.Description("Target remote node"),
		),
		mcpkit.WithNumber("gpu",
			mcpkit.Description("GPU index"),
		),
		mcpkit.WithObject("flags",
			mcpkit.Description("Extra llama-server flags"),
			mcpkit.AdditionalProperties(true),
		),
	), r.toolLoad)
	r.server.AddTool(mcpkit.NewTool("nollama_unload",
		mcpkit.WithDescription("Unload a model from a local or remote node"),
		mcpkit.WithString("model",
			mcpkit.Required(),
			mcpkit.Description("Model name to unload"),
		),
		mcpkit.WithString("node",
			mcpkit.Description("Target remote node"),
		),
	), r.toolUnload)
	r.server.AddTool(mcpkit.NewTool("nollama_models",
		mcpkit.WithDescription("List GGUFs on a local or remote node"),
		mcpkit.WithString("node",
			mcpkit.Description("Filter to a specific remote node"),
		),
	), r.toolModels)
	r.server.AddTool(mcpkit.NewTool("nollama_pull",
		mcpkit.WithDescription("Pull a HuggingFace model on a local or remote node"),
		mcpkit.WithString("spec",
			mcpkit.Required(),
			mcpkit.Description("HuggingFace spec, e.g. unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S"),
		),
		mcpkit.WithString("node",
			mcpkit.Description("Target remote node"),
		),
	), r.toolPull)
	r.server.AddTool(mcpkit.NewTool("nollama_inspect",
		mcpkit.WithDescription("Inspect a GGUF and show hardware guidance"),
		mcpkit.WithString("model",
			mcpkit.Required(),
			mcpkit.Description("GGUF filename or fuzzy match"),
		),
	), r.toolInspect)
	r.server.AddTool(mcpkit.NewTool("nollama_runtimes",
		mcpkit.WithDescription("List installed llama-server runtimes"),
	), r.toolRuntimes)
	r.server.AddTool(mcpkit.NewTool("nollama_rm",
		mcpkit.WithDescription("Remove a local or remote model file"),
		mcpkit.WithString("model",
			mcpkit.Required(),
			mcpkit.Description("Model name to remove"),
		),
		mcpkit.WithString("node",
			mcpkit.Description("Target remote node"),
		),
	), r.toolRm)
}

func (r *Runner) toolStatus(ctx context.Context, req mcpkit.CallToolRequest) (*mcpkit.CallToolResult, error) {
	node := strings.TrimSpace(argString(req.GetArguments(), "node"))
	text, err := r.statusText(ctx, node)
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	return mcpkit.NewToolResultText(text), nil
}

func (r *Runner) toolLoad(ctx context.Context, req mcpkit.CallToolRequest) (*mcpkit.CallToolResult, error) {
	args := req.GetArguments()
	modelName := strings.TrimSpace(argString(args, "model"))
	if modelName == "" {
		return mcpkit.NewToolResultError("model is required"), nil
	}

	node := strings.TrimSpace(argString(args, "node"))
	gpu, gpuOK := argInt(args, "gpu")
	flags, _ := argObject(args, "flags")
	client, err := r.clientForNode(node)
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	reqBody := federation.LoadRequest{Model: filepath.Base(modelName), Flags: flags}
	if gpuOK {
		reqBody.GPU = &gpu
	}
	resp, err := client.Load(reqBody)
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	target := "local"
	if node != "" {
		target = node
	}
	text := fmt.Sprintf("Loaded model %s on %s (port %d, PID %d, device %s)", resp.Model, target, resp.Port, resp.PID, resp.Device)
	return mcpkit.NewToolResultText(text), nil
}

func (r *Runner) toolUnload(ctx context.Context, req mcpkit.CallToolRequest) (*mcpkit.CallToolResult, error) {
	args := req.GetArguments()
	modelName := strings.TrimSpace(argString(args, "model"))
	if modelName == "" {
		return mcpkit.NewToolResultError("model is required"), nil
	}
	node := strings.TrimSpace(argString(args, "node"))
	client, err := r.clientForNode(node)
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	if err := client.Unload(modelName); err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	target := "local"
	if node != "" {
		target = node
	}
	return mcpkit.NewToolResultText(fmt.Sprintf("Unloaded model %s on %s", filepath.Base(modelName), target)), nil
}

func (r *Runner) toolModels(ctx context.Context, req mcpkit.CallToolRequest) (*mcpkit.CallToolResult, error) {
	node := strings.TrimSpace(argString(req.GetArguments(), "node"))
	client, err := r.clientForNode(node)
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	resp, err := client.Models()
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	if len(resp.Models) == 0 {
		target := "local"
		if node != "" {
			target = node
		}
		return mcpkit.NewToolResultText(fmt.Sprintf("No GGUF models found on %s", target)), nil
	}
	var b strings.Builder
	w := tabwriter.NewWriter(&b, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "MODEL\tSIZE\tARCH\tQUANT\tCONTEXT")
	for _, m := range resp.Models {
		name := strings.TrimSuffix(m.Name, ".gguf")
		arch := m.Arch
		if arch == "" {
			arch = "-"
		}
		quant := m.Quant
		if quant == "" {
			quant = "-"
		}
		ctxLen := "-"
		if m.ContextLength > 0 {
			ctxLen = formatContextLength(m.ContextLength)
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", name, m.SizeHuman, arch, quant, ctxLen)
	}
	_ = w.Flush()
	return mcpkit.NewToolResultText(b.String()), nil
}

func (r *Runner) toolPull(ctx context.Context, req mcpkit.CallToolRequest) (*mcpkit.CallToolResult, error) {
	args := req.GetArguments()
	spec := strings.TrimSpace(argString(args, "spec"))
	if spec == "" {
		return mcpkit.NewToolResultError("spec is required"), nil
	}
	node := strings.TrimSpace(argString(args, "node"))
	client, err := r.clientForNode(node)
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	resp, err := client.Pull(spec)
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	target := "local"
	if node != "" {
		target = node
	}
	return mcpkit.NewToolResultText(fmt.Sprintf("Pulled %s on %s (%s)", resp.Filename, target, humanBytes(resp.Size))), nil
}

func (r *Runner) toolInspect(ctx context.Context, req mcpkit.CallToolRequest) (*mcpkit.CallToolResult, error) {
	modelName := strings.TrimSpace(argString(req.GetArguments(), "model"))
	if modelName == "" {
		return mcpkit.NewToolResultError("model is required"), nil
	}

	path, err := r.resolveModelPath(modelName)
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}

	meta, err := model.ParseGGUF(path)
	if err != nil {
		return mcpkit.NewToolResultError(fmt.Sprintf("failed to parse GGUF: %v", err)), nil
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Model:   %s\n", strings.TrimSuffix(filepath.Base(path), ".gguf"))
	fmt.Fprintf(&b, "Arch:    %s\n", meta.ArchDisplayName())
	fmt.Fprintf(&b, "Quant:   %s\n", meta.QuantDisplayName(filepath.Base(path)))
	fmt.Fprintf(&b, "Size:    %.1f GB\n", meta.FileSizeGB())
	if meta.ContextLength > 0 {
		fmt.Fprintf(&b, "Context: %s (embedded)\n", formatContextLength(meta.ContextLength))
	}
	if meta.Name != "" {
		fmt.Fprintf(&b, "Name:    %s\n", meta.Name)
	}

	inv, err := hardware.Detect()
	if err == nil && inv != nil {
		fmt.Fprintln(&b, "")
		fmt.Fprintln(&b, "Hardware:")
		for _, gpu := range inv.GPUs {
			fmt.Fprintf(&b, "  GPU %d: %s — %.0f MiB free of %.0f MiB\n",
				gpu.Index, gpu.DisplayName(), float64(gpu.VRAMFree), float64(gpu.VRAMTotal))
		}
		fmt.Fprintf(&b, "  CPU: %s, %d threads, %.0f GB free RAM\n",
			inv.CPU.ModelName, inv.CPU.Threads, inv.CPU.RAMFreeGB())
	}

	return mcpkit.NewToolResultText(b.String()), nil
}

func (r *Runner) toolRuntimes(ctx context.Context, req mcpkit.CallToolRequest) (*mcpkit.CallToolResult, error) {
	mgr := runtime.NewManager()
	runtimes, err := mgr.List()
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	if len(runtimes) == 0 {
		return mcpkit.NewToolResultText("No runtimes installed"), nil
	}

	var b strings.Builder
	w := tabwriter.NewWriter(&b, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "NAME\tPATH\tSOURCE\tACTIVE")
	for _, rt := range runtimes {
		active := "no"
		if rt.Active {
			active = "yes"
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", rt.Name, rt.Path, rt.Source, active)
	}
	_ = w.Flush()
	return mcpkit.NewToolResultText(b.String()), nil
}

func (r *Runner) toolRm(ctx context.Context, req mcpkit.CallToolRequest) (*mcpkit.CallToolResult, error) {
	args := req.GetArguments()
	modelName := strings.TrimSpace(argString(args, "model"))
	if modelName == "" {
		return mcpkit.NewToolResultError("model is required"), nil
	}
	node := strings.TrimSpace(argString(args, "node"))
	client, err := r.clientForNode(node)
	if err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	if err := client.Rm(modelName); err != nil {
		return mcpkit.NewToolResultError(err.Error()), nil
	}
	if node != "" {
		return mcpkit.NewToolResultText(fmt.Sprintf("Removed %s from %s", filepath.Base(modelName), node)), nil
	}
	return mcpkit.NewToolResultText(fmt.Sprintf("Removed %s", filepath.Base(modelName))), nil
}

func (r *Runner) clientForNode(node string) (*federation.Client, error) {
	node = strings.TrimSpace(node)
	if node == "" {
		return federation.NewClient(r.localBaseURL), nil
	}
	registry, cfgRemotes, err := r.loadRemotes()
	if err != nil {
		return nil, err
	}
	merged := federation.MergeRemotes(registry, cfgRemotes)
	baseURL, ok := merged[node]
	if !ok {
		return nil, fmt.Errorf("remote node %q not found in registry", node)
	}
	return federation.NewClient(baseURL), nil
}

func (r *Runner) loadRemotes() (*federation.RemoteRegistry, map[string]config.Remote, error) {
	registry, err := federation.LoadRegistry(r.registryPath)
	if err != nil {
		return nil, nil, err
	}
	cfgRemotes := map[string]config.Remote{}
	if r.cfg != nil && r.cfg.Remotes != nil {
		cfgRemotes = r.cfg.Remotes
	}
	return registry, cfgRemotes, nil
}

func (r *Runner) statusText(ctx context.Context, node string) (string, error) {
	node = strings.TrimSpace(node)
	if node != "" {
		client, err := r.clientForNode(node)
		if err != nil {
			return "", err
		}
		return renderFleetStatus(map[string]statusResult{
			node: {label: node, resp: mustStatus(client)},
		}), nil
	}

	registry, cfgRemotes, err := r.loadRemotes()
	if err != nil {
		return "", err
	}
	merged := federation.MergeRemotes(registry, cfgRemotes)

	results := make(map[string]statusResult, len(merged)+1)
	results["local"] = statusResult{label: "local", resp: mustStatus(federation.NewClient(r.localBaseURL))}

	var mu sync.Mutex
	var wg sync.WaitGroup
	for name, baseURL := range merged {
		name := name
		baseURL := baseURL
		wg.Add(1)
		go func() {
			defer wg.Done()
			client := federation.NewClient(baseURL)
			client.HTTPClient = &http.Client{Timeout: 5 * time.Second}
			resp, err := client.Status()
			res := statusResult{label: name, resp: resp, err: err}
			mu.Lock()
			results[name] = res
			mu.Unlock()
		}()
	}
	wg.Wait()

	return renderFleetStatus(results), nil
}

type statusResult struct {
	label string
	resp  *federation.StatusResponse
	err   error
}

func mustStatus(client *federation.Client) *federation.StatusResponse {
	if client == nil {
		return nil
	}
	resp, err := client.Status()
	if err != nil {
		return nil
	}
	return resp
}

func renderFleetStatus(results map[string]statusResult) string {
	names := make([]string, 0, len(results))
	for name := range results {
		names = append(names, name)
	}
	sort.Strings(names)

	var b strings.Builder
	w := tabwriter.NewWriter(&b, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "NODE\tGPU\tMODEL\tVRAM\tPORT\tPID\tUPTIME")
	for _, name := range names {
		res := results[name]
		if res.err != nil || res.resp == nil {
			fmt.Fprintf(w, "%s\toffline\t-\t-\t-\t-\t-\n", res.label)
			continue
		}
		gpu := "-"
		if len(res.resp.Node.GPUs) > 0 {
			gpu = res.resp.Node.GPUs[0].Name
		}
		if len(res.resp.Models) == 0 {
			fmt.Fprintf(w, "%s\t%s\t-\t-\t-\t-\t-\n", res.label, gpu)
			continue
		}
		for i, model := range res.resp.Models {
			label := res.label
			if i > 0 {
				label = ""
			}
			fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%d\t%d\t%s\n",
				label,
				gpu,
				model.Name,
				formatVRAM(res.resp.Node.GPUs),
				model.Port,
				model.PID,
				formatUptime(model.UptimeSeconds),
			)
		}
	}
	_ = w.Flush()
	return b.String()
}

func formatVRAM(gpus []federation.StatusGPU) string {
	if len(gpus) == 0 {
		return "-"
	}
	gpu := gpus[0]
	return fmt.Sprintf("%.1f/%.1fGB", float64(gpu.VRAMFreeMB)/1024.0, float64(gpu.VRAMTotalMB)/1024.0)
}

func formatUptime(seconds int64) string {
	if seconds <= 0 {
		return "-"
	}
	d := time.Duration(seconds) * time.Second
	days := d / (24 * time.Hour)
	d -= days * 24 * time.Hour
	hours := d / time.Hour
	if days > 0 {
		return fmt.Sprintf("%dd %dh", days, hours)
	}
	minutes := d / time.Minute
	if hours > 0 {
		return fmt.Sprintf("%dh %dm", hours, minutes)
	}
	return fmt.Sprintf("%dm", minutes)
}

func argString(args map[string]any, key string) string {
	if args == nil {
		return ""
	}
	if v, ok := args[key]; ok {
		switch x := v.(type) {
		case string:
			return x
		case fmt.Stringer:
			return x.String()
		}
	}
	return ""
}

func argInt(args map[string]any, key string) (int, bool) {
	if args == nil {
		return 0, false
	}
	v, ok := args[key]
	if !ok {
		return 0, false
	}
	switch x := v.(type) {
	case int:
		return x, true
	case int64:
		return int(x), true
	case float64:
		return int(x), true
	}
	return 0, false
}

func argObject(args map[string]any, key string) (map[string]any, bool) {
	if args == nil {
		return nil, false
	}
	v, ok := args[key]
	if !ok || v == nil {
		return nil, false
	}
	switch x := v.(type) {
	case map[string]any:
		return x, true
	}
	return nil, false
}

func formatContextLength(ctxLen uint64) string {
	if ctxLen >= 1024*1024 {
		return fmt.Sprintf("%.0fM", float64(ctxLen)/(1024*1024))
	}
	if ctxLen >= 1024 {
		return fmt.Sprintf("%dK", ctxLen/1024)
	}
	return fmt.Sprintf("%d", ctxLen)
}

func humanBytes(size int64) string {
	const (
		gb = 1024 * 1024 * 1024
		mb = 1024 * 1024
	)
	if size >= gb {
		return fmt.Sprintf("%.1f GB", float64(size)/float64(gb))
	}
	if size >= mb {
		return fmt.Sprintf("%.1f MB", float64(size)/float64(mb))
	}
	return fmt.Sprintf("%d B", size)
}

func (r *Runner) resolveModelPath(name string) (string, error) {
	if filepath.IsAbs(name) {
		return name, nil
	}
	dir := ""
	if r.cfg != nil {
		dir = r.cfg.ModelDir
	}
	if dir == "" {
		dir = config.DefaultConfig().ModelDir
	}
	entries, err := model.ScanDir(dir)
	if err != nil {
		return "", err
	}
	names := make([]string, 0, len(entries))
	byName := make(map[string]model.ModelInfo, len(entries))
	for _, m := range entries {
		byName[m.Filename] = m
		names = append(names, m.Filename)
	}
	match := model.FuzzyMatchModel(name, names)
	if match == "" {
		return "", fmt.Errorf("model %q not found in %s", name, dir)
	}
	return byName[match].Path, nil
}
