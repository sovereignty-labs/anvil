package main

import (
	"testing"

	"github.com/sovereignty-labs/anvil/internal/federation"
)

func TestParseDeviceIndex(t *testing.T) {
	cases := []struct {
		gpu         string
		wantIndex   int
		wantBackend string
		wantOK      bool
	}{
		{gpu: "cuda:3", wantIndex: 3, wantBackend: "cuda", wantOK: true},
		{gpu: "rocm:0", wantIndex: 0, wantBackend: "rocm", wantOK: true},
		{gpu: "vulkan:12", wantIndex: 12, wantBackend: "vulkan", wantOK: true},
		{gpu: "cpu", wantOK: false},
		{gpu: "foo:1", wantOK: false},
	}

	for _, tc := range cases {
		idx, backend, ok := parseDeviceIndex(tc.gpu)
		if idx != tc.wantIndex || backend != tc.wantBackend || ok != tc.wantOK {
			t.Fatalf("parseDeviceIndex(%q) = (%d, %q, %t), want (%d, %q, %t)", tc.gpu, idx, backend, ok, tc.wantIndex, tc.wantBackend, tc.wantOK)
		}
	}
}

func TestModelGPUNameAndVRAMMatchBackend(t *testing.T) {
	node := federation.StatusNode{
		GPUs: []federation.StatusGPU{
			{Index: 0, Name: "GeForce RTX 5060 Ti", Backend: "cuda", VRAMTotalMB: 16384, VRAMFreeMB: 512},
			{Index: 0, Name: "AMD Radeon AI PRO R9700", Backend: "rocm", VRAMTotalMB: 15936, VRAMFreeMB: 1024},
			{Index: 1, Name: "Intel Arc", Backend: "vulkan", VRAMTotalMB: 8192, VRAMFreeMB: 2048},
		},
	}

	if got := modelGPUName(node, "rocm:0"); got != "AMD Radeon AI PRO R9700" {
		t.Fatalf("modelGPUName(rocm:0) = %q, want AMD Radeon AI PRO R9700", got)
	}
	if got := modelVRAM(node, "rocm:0"); got != "14.6/15.6GB" {
		t.Fatalf("modelVRAM(rocm:0) = %q, want 14.6/15.6GB", got)
	}
	if got := modelGPUName(node, "vulkan:1"); got != "Intel Arc" {
		t.Fatalf("modelGPUName(vulkan:1) = %q, want Intel Arc", got)
	}
	if got := modelVRAM(node, "cpu"); got != "—" {
		t.Fatalf("modelVRAM(cpu) = %q, want —", got)
	}
}

func TestModelGPUNameBackCompatWithEmptyBackend(t *testing.T) {
	node := federation.StatusNode{
		GPUs: []federation.StatusGPU{{
			Index:       0,
			Name:        "Legacy AMD GPU",
			VRAMTotalMB: 15936,
			VRAMFreeMB:  1024,
		}},
	}

	if got := modelGPUName(node, "rocm:0"); got != "Legacy AMD GPU" {
		t.Fatalf("modelGPUName(rocm:0) = %q, want Legacy AMD GPU", got)
	}
	if got := modelVRAM(node, "rocm:0"); got != "14.6/15.6GB" {
		t.Fatalf("modelVRAM(rocm:0) = %q, want 14.6/15.6GB", got)
	}
}
