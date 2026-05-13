// Package hardware provides GPU and CPU detection for nollama.
package hardware

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

type GPU struct {
	Index      int
	Name       string
	VRAMTotal  uint64 // MiB
	VRAMFree   uint64 // MiB
	VRAMUsed   uint64 // MiB
	PCIBusID   string
	ComputeCap string
}

func (g *GPU) VRAMTotalGB() float64 { return float64(g.VRAMTotal) / 1024.0 }
func (g *GPU) VRAMFreeGB() float64  { return float64(g.VRAMFree) / 1024.0 }
func (g *GPU) VRAMUsedGB() float64  { return float64(g.VRAMUsed) / 1024.0 }
func (g *GPU) DisplayName() string  { return strings.TrimPrefix(g.Name, "NVIDIA ") }

type CPU struct {
	ModelName  string
	Cores      int
	Threads    int
	RAMTotalMB uint64
	RAMFreeMB  uint64
}

func (c *CPU) RAMTotalGB() float64 { return float64(c.RAMTotalMB) / 1024.0 }
func (c *CPU) RAMFreeGB() float64  { return float64(c.RAMFreeMB) / 1024.0 }

type Inventory struct {
	GPUs []GPU
	CPU  CPU
}

func (i *Inventory) TotalRAMGB() float64 { return i.CPU.RAMTotalGB() }

// DetectGPUs enumerates NVIDIA GPUs via nvidia-smi.
// Returns nil (not error) if nvidia-smi is not found.
func DetectGPUs() ([]GPU, error) {
	smiPath, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil, nil
	}
	cmd := exec.Command(smiPath,
		"--query-gpu=index,name,memory.total,memory.free,memory.used,pci.bus_id,compute_cap",
		"--format=csv,noheader,nounits",
	)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi failed: %w", err)
	}
	var gpus []GPU
	scanner := bufio.NewScanner(strings.NewReader(string(out)))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" { continue }
		parts := strings.Split(line, ", ")
		if len(parts) < 7 { continue }
		idx, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		vramTotal, _ := strconv.ParseUint(strings.TrimSpace(parts[2]), 10, 64)
		vramFree, _ := strconv.ParseUint(strings.TrimSpace(parts[3]), 10, 64)
		vramUsed, _ := strconv.ParseUint(strings.TrimSpace(parts[4]), 10, 64)
		gpus = append(gpus, GPU{
			Index: idx, Name: strings.TrimSpace(parts[1]),
			VRAMTotal: vramTotal, VRAMFree: vramFree, VRAMUsed: vramUsed,
			PCIBusID: strings.TrimSpace(parts[5]), ComputeCap: strings.TrimSpace(parts[6]),
		})
	}
	return gpus, nil
}

// DetectCPU reads CPU and memory info from /proc.
func DetectCPU() (CPU, error) {
	cpu := CPU{}
	if f, err := os.Open("/proc/cpuinfo"); err == nil {
		defer f.Close()
		scanner := bufio.NewScanner(f)
		physicalIDs := make(map[string]bool)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "model name") && cpu.ModelName == "" {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 { cpu.ModelName = strings.TrimSpace(parts[1]) }
			}
			if strings.HasPrefix(line, "processor") { cpu.Threads++ }
			if strings.HasPrefix(line, "physical id") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 { physicalIDs[strings.TrimSpace(parts[1])] = true }
			}
			if strings.HasPrefix(line, "cpu cores") && cpu.Cores == 0 {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 { cpu.Cores, _ = strconv.Atoi(strings.TrimSpace(parts[1])) }
			}
		}
		if len(physicalIDs) > 1 { cpu.Cores *= len(physicalIDs) }
	}
	if f, err := os.Open("/proc/meminfo"); err == nil {
		defer f.Close()
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "MemTotal:") { cpu.RAMTotalMB = parseMemKB(line) / 1024 }
			if strings.HasPrefix(line, "MemAvailable:") { cpu.RAMFreeMB = parseMemKB(line) / 1024 }
		}
	}
	return cpu, nil
}

func parseMemKB(line string) uint64 {
	parts := strings.Fields(line)
	if len(parts) < 2 { return 0 }
	val, _ := strconv.ParseUint(parts[1], 10, 64)
	return val
}

func Detect() (*Inventory, error) {
	gpus, err := DetectGPUs()
	if err != nil { return nil, fmt.Errorf("GPU detection: %w", err) }
	cpu, err := DetectCPU()
	if err != nil { return nil, fmt.Errorf("CPU detection: %w", err) }
	return &Inventory{GPUs: gpus, CPU: cpu}, nil
}