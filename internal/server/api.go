package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/sovereignty-labs/nollama/internal/config"
	"github.com/sovereignty-labs/nollama/internal/hardware"
	"github.com/sovereignty-labs/nollama/internal/model"
	"github.com/sovereignty-labs/nollama/internal/process"
	"github.com/sovereignty-labs/nollama/internal/pull"
	runtimemgr "github.com/sovereignty-labs/nollama/internal/runtime"
)

type apiErrorResponse struct {
	Error string `json:"error"`
}

type loadRequest struct {
	Model string         `json:"model"`
	GPU   *int           `json:"gpu,omitempty"`
	CPU   bool           `json:"cpu,omitempty"`
	Flags map[string]any `json:"flags,omitempty"`
	Swap  bool           `json:"swap,omitempty"`
	Port  int            `json:"port,omitempty"`
	Alias string         `json:"alias,omitempty"`
}

type loadResponse struct {
	Status string `json:"status"`
	Model  string `json:"model"`
	Port   int    `json:"port"`
	Device string `json:"device"`
	PID    int    `json:"pid"`
}

type unloadRequest struct {
	Model string `json:"model"`
}

type unloadResponse struct {
	Status string `json:"status"`
}

type modelSummary struct {
	Name          string `json:"name"`
	SizeBytes     int64  `json:"size_bytes"`
	SizeHuman     string `json:"size_human"`
	Arch          string `json:"arch"`
	Quant         string `json:"quant"`
	ContextLength uint64 `json:"context_length"`
}

type modelsResponse struct {
	Models []modelSummary `json:"models"`
}

type pullRequest struct {
	Spec string `json:"spec"`
}

type pullResponse struct {
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
}

type rmRequest struct {
	Model string `json:"model"`
}

type rmResponse struct {
	Filename string `json:"filename"`
	Deleted  bool   `json:"deleted"`
}

type uploadConflictResponse struct {
	Error    string `json:"error"`
	Filename string `json:"filename"`
}

type uploadResponse struct {
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
	SHA256   string `json:"sha256"`
}

func (s *Server) handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w, http.MethodPost)
		return
	}

	filename := strings.TrimSpace(r.Header.Get("X-Filename"))
	if filename == "" {
		writeAPIError(w, http.StatusBadRequest, "X-Filename is required")
		return
	}
	if filepath.Base(filename) != filename {
		writeAPIError(w, http.StatusBadRequest, "invalid filename")
		return
	}

	sizeHeader := strings.TrimSpace(r.Header.Get("X-Content-Length"))
	if sizeHeader == "" {
		writeAPIError(w, http.StatusBadRequest, "X-Content-Length is required")
		return
	}
	expectedSize, err := strconv.ParseInt(sizeHeader, 10, 64)
	if err != nil || expectedSize < 0 {
		writeAPIError(w, http.StatusBadRequest, "invalid X-Content-Length")
		return
	}

	finalPath := s.cfg.ModelPath(filename)
	if info, err := os.Stat(finalPath); err == nil && info.Size() == expectedSize {
		writeJSON(w, http.StatusConflict, uploadConflictResponse{
			Error:    "already exists",
			Filename: filename,
		})
		return
	} else if err != nil && !os.IsNotExist(err) {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}

	if err := os.MkdirAll(filepath.Dir(finalPath), 0o755); err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}

	partialPath := finalPath + ".partial"
	partial, err := os.Create(partialPath)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}
	cleanup := true
	defer func() {
		_ = partial.Close()
		if cleanup {
			_ = os.Remove(partialPath)
		}
	}()

	hasher := sha256.New()
	written, err := io.Copy(partial, io.TeeReader(r.Body, hasher))
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, fmt.Sprintf("upload failed: %v", err))
		return
	}
	if written != expectedSize {
		writeAPIError(w, http.StatusBadRequest, "content length mismatch")
		return
	}

	if shaHeader := strings.TrimSpace(r.Header.Get("X-SHA256")); shaHeader != "" {
		actualSHA := hex.EncodeToString(hasher.Sum(nil))
		if !strings.EqualFold(actualSHA, shaHeader) {
			writeAPIError(w, http.StatusBadRequest, "sha256 mismatch")
			return
		}
	}

	if err := partial.Close(); err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := os.Rename(partialPath, finalPath); err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}

	cleanup = false
	writeJSON(w, http.StatusOK, uploadResponse{
		Filename: filename,
		Size:     written,
		SHA256:   hex.EncodeToString(hasher.Sum(nil)),
	})
}

func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w, http.MethodPost)
		return
	}

	var req pullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	req.Spec = strings.TrimSpace(req.Spec)
	if req.Spec == "" {
		writeAPIError(w, http.StatusBadRequest, "spec is required")
		return
	}

	if _, err := pull.ParseSpec(req.Spec); err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error())
		return
	}

	modelDir := s.cfg.ModelDir
	if modelDir == "" {
		writeAPIError(w, http.StatusInternalServerError, "model directory is required")
		return
	}

	token := os.Getenv("HF_TOKEN")
	resultPath, err := pull.Pull(req.Spec, pull.PullOpts{
		ModelDir: modelDir,
		HFToken:  token,
	})
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error())
		return
	}

	info, err := os.Stat(resultPath)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, fmt.Sprintf("stat downloaded file %s: %v", resultPath, err))
		return
	}

	writeJSON(w, http.StatusOK, pullResponse{
		Filename: filepath.Base(resultPath),
		Size:     info.Size(),
	})
}

func (s *Server) handleRm(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete && r.Method != http.MethodPost {
		writeMethodNotAllowed(w, http.MethodDelete)
		return
	}

	var req rmRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		writeAPIError(w, http.StatusBadRequest, "model is required")
		return
	}

	models, err := model.ScanDir(s.cfg.ModelDir)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}
	available := make([]string, 0, len(models))
	for _, m := range models {
		available = append(available, m.Filename)
	}
	match := model.FuzzyMatchModel(req.Model, available)
	if match == "" {
		writeAPIError(w, http.StatusBadRequest, fmt.Sprintf("model not found: %s", req.Model))
		return
	}

	for _, proc := range s.procMgr.List() {
		if filepath.Base(proc.ModelName) == match || filepath.Base(proc.ModelName) == strings.TrimSuffix(match, ".gguf") {
			writeAPIError(w, http.StatusConflict, fmt.Sprintf("model is currently loaded: %s", match))
			return
		}
	}

	finalPath := s.cfg.ModelPath(match)
	info, err := os.Stat(finalPath)
	if err != nil {
		if os.IsNotExist(err) {
			writeAPIError(w, http.StatusBadRequest, fmt.Sprintf("model not found: %s", req.Model))
			return
		}
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}

	if err := os.Remove(finalPath); err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, rmResponse{
		Filename: match,
		Deleted:  true,
	})
	_ = info
}

func (s *Server) handleLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w, http.MethodPost)
		return
	}

	var req loadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if strings.TrimSpace(req.Model) == "" {
		writeAPIError(w, http.StatusBadRequest, "model is required")
		return
	}

	if s.proxy.HasRouteWithAlias(req.Model, req.Alias) {
		key := req.Alias
		if key == "" {
			key = req.Model
		}
		writeAPIError(w, http.StatusConflict, fmt.Sprintf("model %s is already loaded (route %q)", req.Model, key))
		return
	}

	modelPath := s.cfg.ModelPath(req.Model)
	if _, err := os.Stat(modelPath); err != nil {
		writeAPIError(w, http.StatusBadRequest, fmt.Sprintf("model not found: %s", req.Model))
		return
	}

	meta, err := model.ParseGGUF(modelPath)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, fmt.Sprintf("parse GGUF %s: %v", req.Model, err))
		return
	}

	hw := s.detectHardware()
	if hw == nil {
		writeAPIError(w, http.StatusInternalServerError, "hardware detection failed")
		return
	}

	llamaServer, backend, err := s.resolveLlamaServerPathAndBackend()
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}

	computeInv := hw
	if req.CPU {
		computeInv = cloneInventoryWithoutGPUs(hw)
	} else if req.GPU != nil && *req.GPU >= 0 {
		computeInv, err = inventoryWithGPU(hw, *req.GPU, backend)
		if err != nil {
			writeAPIError(w, http.StatusBadRequest, err.Error())
			return
		}
	}

	modelIndex := s.nextModelIndex()
	result, err := process.ComputeFlags(meta, modelPath, computeInv, llamaServer, modelIndex, backend)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Port > 0 {
		process.OverrideResultPort(result, req.Port)
	}

	passthrough := config.FlagsMapToSlice(req.Flags)
	if alias := strings.TrimSpace(req.Alias); alias != "" {
		passthrough = append(passthrough, "--alias", alias)
	}
	procInfo, err := s.procMgr.Start(result, filepath.Base(req.Model), passthrough)
	if err != nil && req.Swap {
		lru := s.proxy.LRURoute()
		if lru == nil {
			writeAPIError(w, http.StatusInternalServerError, fmt.Sprintf("%v (swap: no models loaded to evict)", err))
			return
		}
		s.logger.Info("swapping out LRU model",
			"model", lru.ModelName,
			"last_request", lru.LastRequest,
			"idle_for", time.Since(lru.LastRequest).Round(time.Second),
			"reason", err.Error(),
		)
		if _, stopErr := s.procMgr.StopByPort(lru.Port); stopErr != nil {
			writeAPIError(w, http.StatusInternalServerError, fmt.Sprintf("evict LRU %s: %v", lru.ModelName, stopErr))
			return
		}
		s.proxy.RemoveRouteByPort(lru.Port)

		procInfo, err = s.procMgr.Start(result, filepath.Base(req.Model), passthrough)
	}
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}
	procInfo.ModelPath = modelPath
	s.proxy.AddRouteWithAlias(req.Model, procInfo.Port, req.Alias)

	writeJSON(w, http.StatusOK, loadResponse{
		Status: "ok",
		Model:  filepath.Base(req.Model),
		Port:   procInfo.Port,
		Device: describeDevice(procInfo.GPUIndex, hw),
		PID:    procInfo.PID,
	})
}

func (s *Server) handleUnload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w, http.MethodPost)
		return
	}

	var req unloadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if strings.TrimSpace(req.Model) == "" {
		writeAPIError(w, http.StatusBadRequest, "model is required")
		return
	}

	stopped, err := s.procMgr.StopByModelName(req.Model)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error())
		return
	}
	for _, proc := range stopped {
		s.proxy.RemoveRouteByPort(proc.Port)
	}

	writeJSON(w, http.StatusOK, unloadResponse{Status: "ok"})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeMethodNotAllowed(w, http.MethodGet)
		return
	}

	models, err := model.ScanDir(s.cfg.ModelDir)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}

	resp := modelsResponse{Models: make([]modelSummary, 0, len(models))}
	for _, m := range models {
		sum := modelSummary{
			Name:      m.Filename,
			SizeBytes: m.SizeBytes,
			SizeHuman: m.SizeHuman(),
		}
		if m.Meta != nil {
			sum.Arch = m.Meta.Architecture
			sum.Quant = m.Meta.QuantDisplayName(m.Filename)
			sum.ContextLength = m.Meta.ContextLength
		}
		resp.Models = append(resp.Models, sum)
	}

	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) resolveLlamaServerPath() (string, error) {
	path, _, err := s.resolveLlamaServerPathAndBackend()
	return path, err
}

func (s *Server) resolveLlamaServerPathAndBackend() (string, runtimemgr.BuildBackend, error) {
	if s.cfg != nil && s.cfg.LlamaServer != "" {
		return s.cfg.LlamaServer, runtimemgr.BuildBackendCUDA, nil
	}

	mgr := runtimemgr.NewManager()
	activeName, err := mgr.ActiveName()
	if err != nil {
		return "", runtimemgr.BuildBackendCUDA, err
	}
	path, err := mgr.Resolve()
	if err != nil {
		return "", runtimemgr.BuildBackendCUDA, err
	}
	return path, mgr.RuntimeBackend(activeName), nil
}

func (s *Server) nextModelIndex() int {
	port := parseBindPort(s.cfg.Bind)
	if port < 11434 {
		port = 11434
	}
	port++

	used := make(map[int]bool)
	for _, proc := range s.procMgr.List() {
		used[proc.Port] = true
	}
	for used[port] {
		port++
	}
	return port - 11434
}

func parseBindPort(bind string) int {
	_, portStr, err := net.SplitHostPort(bind)
	if err != nil {
		parts := strings.Split(bind, ":")
		if len(parts) == 0 {
			return 0
		}
		portStr = parts[len(parts)-1]
	}
	port, _ := strconv.Atoi(portStr)
	return port
}

func cloneInventoryWithoutGPUs(inv *hardware.Inventory) *hardware.Inventory {
	if inv == nil {
		return nil
	}
	clone := *inv
	clone.GPUs = nil
	clone.VulkanGPUs = nil
	return &clone
}

func inventoryWithGPU(inv *hardware.Inventory, gpuIndex int, backend runtimemgr.BuildBackend) (*hardware.Inventory, error) {
	if inv == nil {
		return nil, fmt.Errorf("hardware inventory is nil")
	}
	switch backend {
	case runtimemgr.BuildBackendVulkan:
		for _, gpu := range inv.VulkanGPUs {
			if gpu.Index == gpuIndex {
				clone := *inv
				clone.GPUs = nil
				clone.VulkanGPUs = []hardware.VulkanGPU{gpu}
				return &clone, nil
			}
		}
	default:
		for _, gpu := range inv.GPUs {
			if gpu.Index == gpuIndex {
				clone := *inv
				clone.GPUs = []hardware.GPU{gpu}
				clone.VulkanGPUs = nil
				return &clone, nil
			}
		}
	}
	return nil, fmt.Errorf("GPU %d not available", gpuIndex)
}

func describeDevice(gpuIndex string, inv *hardware.Inventory) string {
	if gpuIndex == "" || gpuIndex == "cpu" {
		return "cpu"
	}
	if !strings.HasPrefix(gpuIndex, "cuda:") && !strings.HasPrefix(gpuIndex, "vulkan:") {
		return gpuIndex
	}
	idxStr := strings.TrimPrefix(strings.TrimPrefix(gpuIndex, "cuda:"), "vulkan:")
	idx, err := strconv.Atoi(idxStr)
	if err != nil {
		return gpuIndex
	}
	if inv != nil {
		switch {
		case strings.HasPrefix(gpuIndex, "vulkan:"):
			for _, gpu := range inv.VulkanGPUs {
				if gpu.Index == idx {
					return fmt.Sprintf("GPU %d (%s)", gpu.Index, gpu.Name)
				}
			}
		default:
			for _, gpu := range inv.GPUs {
				if gpu.Index == idx {
					return fmt.Sprintf("GPU %d (%s)", gpu.Index, gpu.DisplayName())
				}
			}
		}
	}
	return fmt.Sprintf("GPU %d", idx)
}

func writeMethodNotAllowed(w http.ResponseWriter, allowed string) {
	w.Header().Set("Allow", allowed)
	writeAPIError(w, http.StatusMethodNotAllowed, "method not allowed")
}

func writeAPIError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, apiErrorResponse{Error: msg})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
