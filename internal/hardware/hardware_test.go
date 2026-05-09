package hardware

import "testing"

func TestDetectCPU(t *testing.T) {
	cpu, err := DetectCPU()
	if err != nil { t.Fatal(err) }
	if cpu.Threads == 0 { t.Error("expected at least 1 thread") }
	if cpu.RAMTotalMB == 0 { t.Error("expected non-zero RAM") }
	t.Logf("CPU: %s, %d cores, %d threads, %.0f GB RAM (%.0f GB free)",
		cpu.ModelName, cpu.Cores, cpu.Threads, cpu.RAMTotalGB(), cpu.RAMFreeGB())
}

func TestDetectGPUs(t *testing.T) {
	gpus, err := DetectGPUs()
	if err != nil { t.Fatal(err) }
	t.Logf("GPUs found: %d", len(gpus))
}

func TestDetect(t *testing.T) {
	inv, err := Detect()
	if err != nil { t.Fatal(err) }
	if inv == nil { t.Fatal("expected non-nil inventory") }
	t.Logf("Inventory: %d GPUs, %d threads, %.0f GB RAM",
		len(inv.GPUs), inv.CPU.Threads, inv.CPU.RAMTotalGB())
}