package hardware

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
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

func TestParseROCmSMICSVWithOneAMDGPU(t *testing.T) {
	out := `
GPU,Metric,Value
GPU[0],Device Name,AMD Radeon RX 7900 XTX
GPU[0],VRAM Total Memory (B),25769803776
GPU[0],VRAM Total Used Memory (B),8388608
`

	gpus := parseROCmSMICSV(out)
	if len(gpus) != 1 {
		t.Fatalf("expected 1 ROCm GPU, got %d", len(gpus))
	}
	gpu := gpus[0]
	if gpu.Index != 0 {
		t.Fatalf("Index = %d, want 0", gpu.Index)
	}
	if gpu.Name != "AMD Radeon RX 7900 XTX" {
		t.Fatalf("Name = %q, want AMD Radeon RX 7900 XTX", gpu.Name)
	}
	if gpu.VRAMTotal != 24576 {
		t.Fatalf("VRAMTotal = %d, want 24576", gpu.VRAMTotal)
	}
	if gpu.VRAMUsed != 8 {
		t.Fatalf("VRAMUsed = %d, want 8", gpu.VRAMUsed)
	}
	if gpu.VRAMFree != 24568 {
		t.Fatalf("VRAMFree = %d, want 24568", gpu.VRAMFree)
	}
}

func TestDetectROCmGPUsMissingReturnsEmpty(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	gpus, err := detectROCmGPUs()
	if err != nil {
		t.Fatalf("detectROCmGPUs error: %v", err)
	}
	if len(gpus) != 0 {
		t.Fatalf("expected no ROCm GPUs when rocm-smi is absent, got %d", len(gpus))
	}
}

func TestDetectFallsBackToROCmWhenNVIDIANoGPUs(t *testing.T) {
	dir := t.TempDir()
	writeExecutableScript(t, dir, "nvidia-smi", "#!/bin/sh\nexit 0\n")
	writeExecutableScript(t, dir, "rocm-smi", `#!/bin/sh
printf '%s\n' \
  'GPU,Metric,Value' \
  'GPU[0],Device Name,AMD Radeon RX 7900 XTX' \
  'GPU[0],VRAM Total Memory (B),25769803776' \
  'GPU[0],VRAM Total Used Memory (B),8388608'
`)
	t.Setenv("PATH", dir)

	inv, err := Detect()
	if err != nil {
		t.Fatalf("Detect error: %v", err)
	}
	if len(inv.GPUs) != 1 {
		t.Fatalf("expected 1 ROCm GPU from fallback, got %d", len(inv.GPUs))
	}
	gpu := inv.GPUs[0]
	if gpu.Name != "AMD Radeon RX 7900 XTX" {
		t.Fatalf("unexpected GPU name: %+v", gpu)
	}
	if gpu.VRAMTotal != 24576 || gpu.VRAMUsed != 8 || gpu.VRAMFree != 24568 {
		t.Fatalf("unexpected GPU VRAM values: %+v", gpu)
	}
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

func TestScanSysfsAMDVRAM(t *testing.T) {
	root := t.TempDir()
	createSysfsGPU(t, root, 0, "0x10de", 6390*1024*1024, 4000*1024*1024)
	createSysfsGPU(t, root, 2, "0x1002", 34208743424, 59953152)

	got := scanSysfsAMDVRAMAt(root)
	if len(got) != 1 {
		t.Fatalf("expected 1 AMD sysfs entry, got %d", len(got))
	}
	if got[0].VendorID != "0x1002" {
		t.Fatalf("VendorID = %q, want 0x1002", got[0].VendorID)
	}
	if got[0].TotalVRAM != 32624 {
		t.Fatalf("TotalVRAM = %d, want 32624", got[0].TotalVRAM)
	}
	if got[0].UsedVRAM != 57 {
		t.Fatalf("UsedVRAM = %d, want 57", got[0].UsedVRAM)
	}
	if got[0].FreeVRAM != 32567 {
		t.Fatalf("FreeVRAM = %d, want 32567", got[0].FreeVRAM)
	}
	if !strings.HasSuffix(got[0].CardPath, "card2") {
		t.Fatalf("CardPath = %q, want .../card2", got[0].CardPath)
	}
}

func TestBackfillVulkanVRAMFromSysfsSingleAMD(t *testing.T) {
	gpus := []VulkanGPU{
		{Index: 0, Name: "NVIDIA GeForce RTX 2060", TotalVRAM: 6390, FreeVRAM: 6390, vendorID: "0x10de"},
		{Index: 1, Name: "AMD Radeon Graphics (RADV GFX1201)", vendorID: "0x1002"},
	}
	sysfs := []sysfsGPUInfo{
		{VendorID: "0x1002", TotalVRAM: 32624, UsedVRAM: 57, FreeVRAM: 32567, CardPath: "/sys/class/drm/card2"},
	}

	backfillVulkanVRAMFromSysfs(gpus, sysfs)

	if gpus[0].TotalVRAM != 6390 {
		t.Fatalf("NVIDIA GPU should not change, got %+v", gpus[0])
	}
	if gpus[1].TotalVRAM != 32624 || gpus[1].FreeVRAM != 32567 {
		t.Fatalf("AMD GPU not backfilled correctly: %+v", gpus[1])
	}
}

func TestBackfillVulkanVRAMFromSysfsNoMatches(t *testing.T) {
	gpus := []VulkanGPU{
		{Index: 0, Name: "AMD Radeon Graphics", vendorID: "0x1002"},
	}
	backfillVulkanVRAMFromSysfs(gpus, nil)
	if gpus[0].TotalVRAM != 0 || gpus[0].FreeVRAM != 0 {
		t.Fatalf("expected no change without sysfs entries, got %+v", gpus[0])
	}
}

func TestBackfillVulkanVRAMFromSysfsDoesNotOverwriteExisting(t *testing.T) {
	gpus := []VulkanGPU{
		{Index: 0, Name: "AMD Radeon Graphics", TotalVRAM: 8192, FreeVRAM: 4096, vendorID: "0x1002"},
	}
	sysfs := []sysfsGPUInfo{
		{VendorID: "0x1002", TotalVRAM: 32624, UsedVRAM: 57, FreeVRAM: 32567, CardPath: "/sys/class/drm/card2"},
	}
	backfillVulkanVRAMFromSysfs(gpus, sysfs)
	if gpus[0].TotalVRAM != 8192 || gpus[0].FreeVRAM != 4096 {
		t.Fatalf("existing VRAM should not be overwritten, got %+v", gpus[0])
	}
}

func createSysfsGPU(t *testing.T, root string, card int, vendor string, totalBytes, usedBytes uint64) {
	t.Helper()
	cardDir := filepath.Join(root, "card"+strconv.Itoa(card), "device")
	if err := os.MkdirAll(cardDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cardDir, "vendor"), []byte(vendor+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cardDir, "mem_info_vram_total"), []byte(strconv.FormatUint(totalBytes, 10)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cardDir, "mem_info_vram_used"), []byte(strconv.FormatUint(usedBytes, 10)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeExecutableScript(t *testing.T, dir, name, body string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}
