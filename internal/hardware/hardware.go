// Package hardware provides GPU and CPU detection for nollama.
package hardware

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
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

type VulkanGPU struct {
	Index      int
	Name       string
	TotalVRAM  uint64 // MiB
	FreeVRAM   uint64 // MiB
	vendorID   string
	deviceType string
}

func (g VulkanGPU) VendorID() string   { return g.vendorID }
func (g VulkanGPU) DeviceType() string { return g.deviceType }

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
	GPUs       []GPU
	ROCmGPUs   []GPU
	VulkanGPUs []VulkanGPU
	CPU        CPU
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
		if line == "" {
			continue
		}
		parts := strings.Split(line, ", ")
		if len(parts) < 7 {
			continue
		}
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

// detectROCmGPUs enumerates AMD GPUs via rocm-smi.
// Returns nil (not error) if rocm-smi is not found.
func detectROCmGPUs() ([]GPU, error) {
	smiPath, err := exec.LookPath("rocm-smi")
	if err != nil {
		return nil, nil
	}
	cmd := exec.Command(smiPath, "--showproductname", "--showmeminfo", "vram", "--csv")
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("rocm-smi failed: %w", err)
	}
	return parseROCmSMICSV(string(out)), nil
}

func parseROCmSMICSV(out string) []GPU {
	reader := csv.NewReader(strings.NewReader(out))
	reader.FieldsPerRecord = -1
	reader.TrimLeadingSpace = true

	header, err := reader.Read()
	if err != nil {
		return nil
	}

	deviceIdx := -1
	totalIdx := -1
	usedIdx := -1
	nameIdx := -1
	for i, col := range header {
		switch normalizeROCmCSVHeader(col) {
		case "device":
			deviceIdx = i
		case "vram total memory (b)":
			totalIdx = i
		case "vram total used memory (b)":
			usedIdx = i
		case "card series":
			nameIdx = i
		}
	}
	if deviceIdx < 0 || totalIdx < 0 || usedIdx < 0 || nameIdx < 0 {
		return nil
	}

	type rocmGPURow struct {
		index    int
		name     string
		totalMiB uint64
		usedMiB  uint64
	}

	rows := make([]rocmGPURow, 0)
	for {
		rec, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil || len(rec) == 0 {
			continue
		}
		if deviceIdx >= len(rec) || totalIdx >= len(rec) || usedIdx >= len(rec) || nameIdx >= len(rec) {
			continue
		}

		idx, ok := parseROCmDeviceIndex(rec[deviceIdx])
		if !ok {
			continue
		}
		totalBytes, ok := parseROCmUint(rec[totalIdx])
		if !ok {
			continue
		}
		usedBytes, ok := parseROCmUint(rec[usedIdx])
		if !ok {
			continue
		}

		rows = append(rows, rocmGPURow{
			index:    idx,
			name:     strings.TrimSpace(rec[nameIdx]),
			totalMiB: bytesToMiB(totalBytes),
			usedMiB:  bytesToMiB(usedBytes),
		})
	}

	sort.Slice(rows, func(i, j int) bool { return rows[i].index < rows[j].index })
	gpus := make([]GPU, 0, len(rows))
	for _, row := range rows {
		free := uint64(0)
		if row.totalMiB > row.usedMiB {
			free = row.totalMiB - row.usedMiB
		}
		gpus = append(gpus, GPU{
			Index:     row.index,
			Name:      row.name,
			VRAMTotal: row.totalMiB,
			VRAMFree:  free,
			VRAMUsed:  row.usedMiB,
		})
	}
	return gpus
}

func normalizeROCmCSVHeader(field string) string {
	field = strings.ToLower(strings.TrimSpace(strings.Trim(field, `"`)))
	field = strings.Join(strings.Fields(field), " ")
	return field
}

func parseROCmDeviceIndex(field string) (int, bool) {
	field = normalizeROCmCSVHeader(field)
	if field == "" {
		return 0, false
	}
	field = strings.TrimPrefix(field, "card")
	if idx, err := strconv.Atoi(field); err == nil {
		return idx, true
	}
	return 0, false
}

func parseROCmUint(value string) (uint64, bool) {
	value = strings.TrimSpace(value)
	value = strings.Trim(value, `"`)
	value = strings.ReplaceAll(value, ",", "")
	if value == "" {
		return 0, false
	}
	fields := strings.Fields(value)
	if len(fields) == 0 {
		return 0, false
	}
	for i, r := range fields[0] {
		if r < '0' || r > '9' {
			fields[0] = fields[0][:i]
			break
		}
	}
	if fields[0] == "" {
		return 0, false
	}
	v, err := strconv.ParseUint(fields[0], 10, 64)
	return v, err == nil
}

var vulkanGPUHeader = regexp.MustCompile(`^GPU([0-9]+):$`)
var drmCardHeader = regexp.MustCompile(`^card([0-9]+)$`)

const sysfsDRMRoot = "/sys/class/drm"

type sysfsGPUInfo struct {
	VendorID  string // e.g. "0x1002"
	TotalVRAM uint64 // MiB
	UsedVRAM  uint64 // MiB
	FreeVRAM  uint64 // MiB
	CardPath  string // e.g. "/sys/class/drm/card2"
}

// DetectVulkanGPUs enumerates Vulkan GPUs via vulkaninfo --summary.
// Returns nil (not error) if vulkaninfo is not found.
func DetectVulkanGPUs() ([]VulkanGPU, error) {
	vulkanPath, err := exec.LookPath("vulkaninfo")
	if err != nil {
		return nil, nil
	}
	cmd := exec.Command(vulkanPath, "--summary")
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("vulkaninfo failed: %w", err)
	}
	gpus := parseVulkanInfoSummary(string(out))
	backfillVulkanVRAMFromSysfs(gpus, scanSysfsAMDVRAM())
	return gpus, nil
}

func parseVulkanInfoSummary(out string) []VulkanGPU {
	var gpus []VulkanGPU
	var current *VulkanGPU

	flush := func() {
		if current != nil {
			gpus = append(gpus, *current)
			current = nil
		}
	}

	scanner := bufio.NewScanner(strings.NewReader(out))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if match := vulkanGPUHeader.FindStringSubmatch(line); len(match) == 2 {
			flush()
			idx, _ := strconv.Atoi(match[1])
			current = &VulkanGPU{Index: idx}
			continue
		}
		if current == nil {
			continue
		}

		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)

		switch key {
		case "vendorID":
			current.vendorID = strings.ToLower(value)
		case "deviceType":
			current.deviceType = value
		case "deviceName":
			current.Name = value
		case "deviceMemory":
			fields := strings.Fields(value)
			if len(fields) == 0 {
				continue
			}
			total, err := strconv.ParseUint(fields[0], 10, 64)
			if err != nil {
				continue
			}
			current.TotalVRAM = total
			current.FreeVRAM = total
		}
	}
	flush()
	return gpus
}

func scanSysfsAMDVRAM() []sysfsGPUInfo {
	return scanSysfsAMDVRAMAt(sysfsDRMRoot)
}

func scanSysfsAMDVRAMAt(drmRoot string) []sysfsGPUInfo {
	matches, err := filepath.Glob(filepath.Join(drmRoot, "card*", "device", "mem_info_vram_total"))
	if err != nil || len(matches) == 0 {
		return nil
	}

	sort.Slice(matches, func(i, j int) bool {
		return sysfsCardIndex(matches[i]) < sysfsCardIndex(matches[j])
	})

	out := make([]sysfsGPUInfo, 0, len(matches))
	for _, totalPath := range matches {
		cardPath := filepath.Dir(filepath.Dir(totalPath))
		devicePath := filepath.Dir(totalPath)
		vendorPath := filepath.Join(devicePath, "vendor")
		vendor, err := readTrimmedFile(vendorPath)
		if err != nil || strings.TrimSpace(vendor) != "0x1002" {
			continue
		}

		totalBytes, err := readUintFile(totalPath)
		if err != nil {
			continue
		}
		usedBytes, err := readUintFile(filepath.Join(devicePath, "mem_info_vram_used"))
		if err != nil {
			continue
		}

		totalMiB := bytesToMiB(totalBytes)
		usedMiB := bytesToMiB(usedBytes)
		freeMiB := uint64(0)
		if totalMiB > usedMiB {
			freeMiB = totalMiB - usedMiB
		}

		out = append(out, sysfsGPUInfo{
			VendorID:  strings.TrimSpace(vendor),
			TotalVRAM: totalMiB,
			UsedVRAM:  usedMiB,
			FreeVRAM:  freeMiB,
			CardPath:  cardPath,
		})
	}
	return out
}

func backfillVulkanVRAMFromSysfs(gpus []VulkanGPU, sysfs []sysfsGPUInfo) {
	if len(gpus) == 0 || len(sysfs) == 0 {
		return
	}

	needs := make([]int, 0, len(gpus))
	for i, gpu := range gpus {
		if gpu.TotalVRAM == 0 && strings.EqualFold(gpu.VendorID(), "0x1002") {
			needs = append(needs, i)
		}
	}
	if len(needs) == 0 {
		return
	}

	for i, gpuIdx := range needs {
		if i >= len(sysfs) {
			break
		}
		info := sysfs[i]
		gpus[gpuIdx].TotalVRAM = info.TotalVRAM
		gpus[gpuIdx].FreeVRAM = info.FreeVRAM
	}
}

func sysfsCardIndex(path string) int {
	base := filepath.Base(filepath.Dir(filepath.Dir(path)))
	match := drmCardHeader.FindStringSubmatch(base)
	if len(match) != 2 {
		return 1<<31 - 1
	}
	idx, err := strconv.Atoi(match[1])
	if err != nil {
		return 1<<31 - 1
	}
	return idx
}

func readTrimmedFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

func readUintFile(path string) (uint64, error) {
	data, err := readTrimmedFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(data, 10, 64)
}

func bytesToMiB(bytes uint64) uint64 {
	const mib = 1024 * 1024
	return bytes / mib
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
				if len(parts) == 2 {
					cpu.ModelName = strings.TrimSpace(parts[1])
				}
			}
			if strings.HasPrefix(line, "processor") {
				cpu.Threads++
			}
			if strings.HasPrefix(line, "physical id") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					physicalIDs[strings.TrimSpace(parts[1])] = true
				}
			}
			if strings.HasPrefix(line, "cpu cores") && cpu.Cores == 0 {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					cpu.Cores, _ = strconv.Atoi(strings.TrimSpace(parts[1]))
				}
			}
		}
		if len(physicalIDs) > 1 {
			cpu.Cores *= len(physicalIDs)
		}
	}
	if f, err := os.Open("/proc/meminfo"); err == nil {
		defer f.Close()
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "MemTotal:") {
				cpu.RAMTotalMB = parseMemKB(line) / 1024
			}
			if strings.HasPrefix(line, "MemAvailable:") {
				cpu.RAMFreeMB = parseMemKB(line) / 1024
			}
		}
	}
	return cpu, nil
}

func parseMemKB(line string) uint64 {
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return 0
	}
	val, _ := strconv.ParseUint(parts[1], 10, 64)
	return val
}

func Detect() (*Inventory, error) {
	gpus, err := DetectGPUs()
	if err != nil {
		return nil, fmt.Errorf("GPU detection: %w", err)
	}
	rocmGPUs, err := detectROCmGPUs()
	if err != nil {
		return nil, fmt.Errorf("ROCm GPU detection: %w", err)
	}
	vulkanGPUs, err := DetectVulkanGPUs()
	if err != nil {
		return nil, fmt.Errorf("Vulkan GPU detection: %w", err)
	}
	cpu, err := DetectCPU()
	if err != nil {
		return nil, fmt.Errorf("CPU detection: %w", err)
	}
	return &Inventory{GPUs: gpus, ROCmGPUs: rocmGPUs, VulkanGPUs: vulkanGPUs, CPU: cpu}, nil
}
