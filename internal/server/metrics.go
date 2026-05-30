package server

import (
	"fmt"
	"net/http"
	"time"

	"github.com/sovereignty-labs/anvil/internal/hardware"
)

// handleMetrics writes Prometheus-format metrics.
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")

	stats := s.proxy.RouteStatsList()
	fmt.Fprintf(w, "# HELP anvil_loaded_models Number of currently loaded models\n")
	fmt.Fprintf(w, "# TYPE anvil_loaded_models gauge\n")
	fmt.Fprintf(w, "anvil_loaded_models %d\n\n", len(stats))

	fmt.Fprintf(w, "# HELP anvil_model_requests_total Total requests proxied per model\n")
	fmt.Fprintf(w, "# TYPE anvil_model_requests_total counter\n")
	for _, st := range stats {
		fmt.Fprintf(w, "anvil_model_requests_total{model=%q} %d\n", st.ModelName, st.RequestCount)
	}
	fmt.Fprintln(w)

	fmt.Fprintf(w, "# HELP anvil_model_last_request_timestamp_seconds Unix timestamp of last request per model\n")
	fmt.Fprintf(w, "# TYPE anvil_model_last_request_timestamp_seconds gauge\n")
	for _, st := range stats {
		fmt.Fprintf(w, "anvil_model_last_request_timestamp_seconds{model=%q} %d\n", st.ModelName, st.LastRequest.Unix())
	}
	fmt.Fprintln(w)

	fmt.Fprintf(w, "# HELP anvil_model_idle_seconds Seconds since last request per model\n")
	fmt.Fprintf(w, "# TYPE anvil_model_idle_seconds gauge\n")
	now := time.Now()
	for _, st := range stats {
		fmt.Fprintf(w, "anvil_model_idle_seconds{model=%q} %.0f\n", st.ModelName, now.Sub(st.LastRequest).Seconds())
	}
	fmt.Fprintln(w)

	inv, err := hardware.Detect()
	if err == nil && inv != nil && len(inv.GPUs) > 0 {
		fmt.Fprintf(w, "# HELP anvil_gpu_vram_total_bytes Total GPU VRAM in bytes\n")
		fmt.Fprintf(w, "# TYPE anvil_gpu_vram_total_bytes gauge\n")
		for _, gpu := range inv.GPUs {
			fmt.Fprintf(w, "anvil_gpu_vram_total_bytes{gpu=\"%d\",name=%q} %d\n",
				gpu.Index, gpu.DisplayName(), int64(gpu.VRAMTotal)*1024*1024)
		}
		fmt.Fprintln(w)

		fmt.Fprintf(w, "# HELP anvil_gpu_vram_used_bytes Used GPU VRAM in bytes\n")
		fmt.Fprintf(w, "# TYPE anvil_gpu_vram_used_bytes gauge\n")
		for _, gpu := range inv.GPUs {
			used := uint64(0)
			if gpu.VRAMTotal > gpu.VRAMFree {
				used = gpu.VRAMTotal - gpu.VRAMFree
			}
			fmt.Fprintf(w, "anvil_gpu_vram_used_bytes{gpu=\"%d\",name=%q} %d\n",
				gpu.Index, gpu.DisplayName(), int64(used)*1024*1024)
		}
		fmt.Fprintln(w)
	}

	if inv != nil {
		fmt.Fprintf(w, "# HELP anvil_system_ram_total_bytes Total system RAM in bytes\n")
		fmt.Fprintf(w, "# TYPE anvil_system_ram_total_bytes gauge\n")
		fmt.Fprintf(w, "anvil_system_ram_total_bytes %d\n\n", int64(inv.CPU.RAMTotalMB)*1024*1024)

		fmt.Fprintf(w, "# HELP anvil_system_ram_free_bytes Free system RAM in bytes\n")
		fmt.Fprintf(w, "# TYPE anvil_system_ram_free_bytes gauge\n")
		fmt.Fprintf(w, "anvil_system_ram_free_bytes %d\n", int64(inv.CPU.RAMFreeMB)*1024*1024)
	}
}
