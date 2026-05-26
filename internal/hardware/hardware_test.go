package hardware

import (
	"testing"
)

func TestDetectCPU(t *testing.T) {
	cpu, err := DetectCPU()
	if err != nil {
		t.Fatal(err)
	}
	if cpu.Threads == 0 {
		t.Error("expected at least 1 thread")
	}
	if cpu.RAMTotalMB == 0 {
		t.Error("expected non-zero RAM")
	}
	t.Logf("CPU: %s, %d cores, %d threads, %.0f GB RAM (%.0f GB free)",
		cpu.ModelName, cpu.Cores, cpu.Threads, cpu.RAMTotalGB(), cpu.RAMFreeGB())
}

func TestDetectGPUs(t *testing.T) {
	gpus, err := DetectGPUs()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("GPUs found: %d", len(gpus))
}

func TestDetect(t *testing.T) {
	inv, err := Detect()
	if err != nil {
		t.Fatal(err)
	}
	if inv == nil {
		t.Fatal("expected non-nil inventory")
	}
	t.Logf("Inventory: %d GPUs, %d threads, %.0f GB RAM",
		len(inv.GPUs), inv.CPU.Threads, inv.CPU.RAMTotalGB())
}

func TestParseVulkanInfoSummaryWithNvidiaAndAMD(t *testing.T) {
	out := `
GPU0:
    apiVersion    = 1.3.283
    driverVersion = 550.78
    vendorID      = 0x10de
    deviceID      = 0x1f08
    deviceType    = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName    = NVIDIA GeForce RTX 2060
    deviceMemory  = 6390 MiB (VRAM)

GPU1:
    apiVersion    = 1.4.304
    driverVersion = 24.2.8
    vendorID      = 0x1002
    deviceID      = 0x7550
    deviceType    = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName    = AMD Radeon Graphics (RADV GFX1201)
    deviceMemory  = 32768 MiB (VRAM)
`
	gpus := parseVulkanInfoSummary(out)
	if len(gpus) != 2 {
		t.Fatalf("expected 2 Vulkan GPUs, got %d", len(gpus))
	}
	if gpus[0].Index != 0 || gpus[0].Name != "NVIDIA GeForce RTX 2060" || gpus[0].TotalVRAM != 6390 {
		t.Fatalf("unexpected GPU0 parse: %+v", gpus[0])
	}
	if gpus[1].Index != 1 || gpus[1].Name != "AMD Radeon Graphics (RADV GFX1201)" || gpus[1].TotalVRAM != 32768 {
		t.Fatalf("unexpected GPU1 parse: %+v", gpus[1])
	}
}

func TestParseVulkanInfoSummaryWithAMDOnly(t *testing.T) {
	out := `
GPU0:
    vendorID      = 0x1002
    deviceType    = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName    = AMD Radeon RX 6800
    deviceMemory  = 16384 MiB (VRAM)
`
	gpus := parseVulkanInfoSummary(out)
	if len(gpus) != 1 {
		t.Fatalf("expected 1 Vulkan GPU, got %d", len(gpus))
	}
	if gpus[0].Name != "AMD Radeon RX 6800" {
		t.Fatalf("unexpected GPU name: %+v", gpus[0])
	}
	if gpus[0].FreeVRAM != 16384 {
		t.Fatalf("expected FreeVRAM 16384, got %+v", gpus[0])
	}
}

func TestDetectVulkanGPUsMissingReturnsEmpty(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	gpus, err := DetectVulkanGPUs()
	if err != nil {
		t.Fatalf("DetectVulkanGPUs error: %v", err)
	}
	if len(gpus) != 0 {
		t.Fatalf("expected no Vulkan GPUs when vulkaninfo is absent, got %d", len(gpus))
	}
}
