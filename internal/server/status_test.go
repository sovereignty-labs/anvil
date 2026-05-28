package server

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestStatusGPUJSONIncludesBackend(t *testing.T) {
	payload := statusGPU{
		Index:       0,
		Name:        "AMD Radeon AI PRO R9700",
		Backend:     "rocm",
		VRAMTotalMB: 15936,
		VRAMFreeMB:  1024,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	if !strings.Contains(string(data), `"backend":"rocm"`) {
		t.Fatalf("expected backend field in %s", data)
	}
}
