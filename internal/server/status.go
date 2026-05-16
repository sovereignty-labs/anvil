package server

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/hirdforge/nollama/internal/hardware"
)

type statusResponse struct {
	Models []statusModel `json:"models"`
	Node   statusNode    `json:"node"`
}

type statusModel struct {
	Name          string `json:"name"`
	Port          int    `json:"port"`
	GPU           string `json:"gpu"`
	PID           int    `json:"pid"`
	UptimeSeconds int64  `json:"uptime_seconds"`
}

type statusNode struct {
	GPUs       []statusGPU `json:"gpus"`
	RAMTotalMB uint64      `json:"ram_total_mb"`
	RAMFreeMB  uint64      `json:"ram_free_mb"`
}

type statusGPU struct {
	Index       int    `json:"index"`
	Name        string `json:"name"`
	VRAMTotalMB uint64 `json:"vram_total_mb"`
	VRAMFreeMB  uint64 `json:"vram_free_mb"`
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	resp := s.buildStatusResponse()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		s.logger.Error("encode status response failed", "error", err)
		http.Error(w, "failed to encode status", http.StatusInternalServerError)
	}
}

func (s *Server) buildStatusResponse() statusResponse {
	resp := statusResponse{
		Models: make([]statusModel, 0),
		Node:   statusNode{},
	}

	hw, err := hardware.Detect()
	if err != nil {
		s.logger.Warn("hardware detection failed for status", "error", err)
	} else if hw != nil {
		resp.Node = statusNode{
			GPUs:       make([]statusGPU, 0, len(hw.GPUs)),
			RAMTotalMB: hw.CPU.RAMTotalMB,
			RAMFreeMB:  hw.CPU.RAMFreeMB,
		}
		for _, gpu := range hw.GPUs {
			resp.Node.GPUs = append(resp.Node.GPUs, statusGPU{
				Index:       gpu.Index,
				Name:        gpu.DisplayName(),
				VRAMTotalMB: gpu.VRAMTotal,
				VRAMFreeMB:  gpu.VRAMFree,
			})
		}
	}

	procs := s.procMgr.List()
	resp.Models = make([]statusModel, 0, len(procs))
	for _, proc := range procs {
		resp.Models = append(resp.Models, statusModel{
			Name:          proc.ModelName,
			Port:          proc.Port,
			GPU:           proc.GPUIndex,
			PID:           proc.PID,
			UptimeSeconds: int64(time.Since(proc.StartTime).Seconds()),
		})
	}

	return resp
}
