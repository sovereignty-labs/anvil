package server

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/hirdforge/nollama/internal/hardware"
	"github.com/hirdforge/nollama/internal/model"
	"github.com/hirdforge/nollama/internal/process"
	runtimemgr "github.com/hirdforge/nollama/internal/runtime"
)

type apiErrorResponse struct {
	Error string `json:"error"`
}

type loadRequest struct {
	Model string         `json:"model"`
	GPU   *int           `json:"gpu,omitempty"`
	CPU   bool           `json:"cpu,omitempty"`
	Flags map[string]any `json:"flags,omitempty"`
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

	llamaServer, err := s.resolveLlamaServerPath()
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}

	computeInv := hw
	if req.CPU {
		computeInv = cloneInventoryWithoutGPUs(hw)
	} else if req.GPU != nil && *req.GPU >= 0 {
		computeInv, err = inventoryWithGPU(hw, *req.GPU)
		if err != nil {
			writeAPIError(w, http.StatusBadRequest, err.Error())
			return
		}
	}

	modelIndex := s.nextModelIndex()
	result, err := process.ComputeFlags(meta, modelPath, computeInv, llamaServer, modelIndex)
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, err.Error())
		return
	}

	passthrough := flagsMapToSlice(req.Flags)
	procInfo, err := s.procMgr.Start(result, filepath.Base(req.Model), passthrough)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, err.Error())
		return
	}
	procInfo.ModelPath = modelPath
	s.proxy.AddRoute(req.Model, procInfo.Port)

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
		s.proxy.RemoveRoute(filepath.Base(proc.ModelName))
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
			sum.Quant = m.Meta.QuantName
			sum.ContextLength = m.Meta.ContextLength
		}
		resp.Models = append(resp.Models, sum)
	}

	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) resolveLlamaServerPath() (string, error) {
	if s.cfg != nil && s.cfg.LlamaServer != "" {
		return s.cfg.LlamaServer, nil
	}

	path, err := runtimemgr.NewManager().Resolve()
	if err != nil {
		return "", err
	}
	return path, nil
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
	return &clone
}

func inventoryWithGPU(inv *hardware.Inventory, gpuIndex int) (*hardware.Inventory, error) {
	if inv == nil {
		return nil, fmt.Errorf("hardware inventory is nil")
	}
	for _, gpu := range inv.GPUs {
		if gpu.Index == gpuIndex {
			clone := *inv
			clone.GPUs = []hardware.GPU{gpu}
			return &clone, nil
		}
	}
	return nil, fmt.Errorf("GPU %d not available", gpuIndex)
}

func describeDevice(gpuIndex string, inv *hardware.Inventory) string {
	if gpuIndex == "" || gpuIndex == "cpu" {
		return "cpu"
	}
	if !strings.HasPrefix(gpuIndex, "cuda:") {
		return gpuIndex
	}
	idxStr := strings.TrimPrefix(gpuIndex, "cuda:")
	idx, err := strconv.Atoi(idxStr)
	if err != nil {
		return gpuIndex
	}
	if inv != nil {
		for _, gpu := range inv.GPUs {
			if gpu.Index == idx {
				return fmt.Sprintf("GPU %d (%s)", gpu.Index, gpu.DisplayName())
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
