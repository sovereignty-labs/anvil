package runtime

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestDetectPlatformWithNvidiaSmi(t *testing.T) {
	tmpDir := t.TempDir()
	script := filepath.Join(tmpDir, "nvidia-smi")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldLookPath := execLookPath
	t.Cleanup(func() {
		execLookPath = oldLookPath
	})

	execLookPath = func(file string) (string, error) {
		if file == "nvidia-smi" {
			return script, nil
		}
		return "", os.ErrNotExist
	}

	platform := DetectPlatform()
	if platform.CUDA == "" {
		t.Fatal("expected CUDA to be detected when nvidia-smi is available")
	}
}

func TestDetectPlatformWithRocmSmi(t *testing.T) {
	tmpDir := t.TempDir()
	script := filepath.Join(tmpDir, "rocm-smi")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldLookPath := execLookPath
	t.Cleanup(func() {
		execLookPath = oldLookPath
	})

	execLookPath = func(file string) (string, error) {
		switch file {
		case "nvidia-smi":
			return "", os.ErrNotExist
		case "rocm-smi":
			return script, nil
		default:
			return "", os.ErrNotExist
		}
	}

	platform := DetectPlatform()
	if platform.ROCm == "" {
		t.Fatal("expected ROCm to be detected when rocm-smi is available")
	}
}

func TestSelectAsset(t *testing.T) {
	assets := []ReleaseAsset{
		{Name: "llama-b9174-bin-ubuntu-x64.zip", Size: 100},
		{Name: "llama-b9174-bin-ubuntu-x64-cuda-cu12.8.0.zip", Size: 200},
		{Name: "llama-b9174-bin-macos-arm64.zip", Size: 300},
	}

	cudaSelected, err := SelectAsset(assets, Platform{OS: "linux", Arch: "amd64", CUDA: "available"})
	if err != nil {
		t.Fatal(err)
	}
	if cudaSelected.Name != "llama-b9174-bin-ubuntu-x64-cuda-cu12.8.0.zip" {
		t.Fatalf("cuda selection = %q", cudaSelected.Name)
	}

	cpuSelected, err := SelectAsset(assets, Platform{OS: "linux", Arch: "amd64"})
	if err != nil {
		t.Fatal(err)
	}
	if cpuSelected.Name != "llama-b9174-bin-ubuntu-x64.zip" {
		t.Fatalf("cpu selection = %q", cpuSelected.Name)
	}

	macSelected, err := SelectAsset(assets, Platform{OS: "darwin", Arch: "arm64"})
	if err != nil {
		t.Fatal(err)
	}
	if macSelected.Name != "llama-b9174-bin-macos-arm64.zip" {
		t.Fatalf("mac selection = %q", macSelected.Name)
	}
}

func TestSelectAssetPrefersCudaOverRocmOnNvidia(t *testing.T) {
	assets := []ReleaseAsset{
		{Name: "llama-b9174-bin-ubuntu-x64.zip", Size: 100},
		{Name: "llama-b9174-bin-ubuntu-x64-rocm-6.2.tar.gz", Size: 500},
		{Name: "llama-b9174-bin-ubuntu-x64-cuda-cu12.8.0.zip", Size: 200},
	}

	selected, err := SelectAsset(assets, Platform{OS: "linux", Arch: "amd64", CUDA: "available"})
	if err != nil {
		t.Fatal(err)
	}
	if selected.Name != "llama-b9174-bin-ubuntu-x64-cuda-cu12.8.0.zip" {
		t.Fatalf("cuda selection = %q", selected.Name)
	}
}

func TestSelectAssetPrefersCpuOverRocmWithoutCuda(t *testing.T) {
	assets := []ReleaseAsset{
		{Name: "llama-b9174-bin-ubuntu-x64.zip", Size: 100},
		{Name: "llama-b9174-bin-ubuntu-x64-rocm-6.2.tar.gz", Size: 500},
	}

	selected, err := SelectAsset(assets, Platform{OS: "linux", Arch: "amd64"})
	if err != nil {
		t.Fatal(err)
	}
	if selected.Name != "llama-b9174-bin-ubuntu-x64.zip" {
		t.Fatalf("cpu selection = %q", selected.Name)
	}
}

func TestSelectAssetPrefersRocmWhenDetected(t *testing.T) {
	assets := []ReleaseAsset{
		{Name: "llama-b9174-bin-ubuntu-x64.zip", Size: 100},
		{Name: "llama-b9174-bin-ubuntu-x64-rocm-6.2.tar.gz", Size: 200},
	}

	selected, err := SelectAsset(assets, Platform{OS: "linux", Arch: "amd64", ROCm: "available"})
	if err != nil {
		t.Fatal(err)
	}
	if selected.Name != "llama-b9174-bin-ubuntu-x64-rocm-6.2.tar.gz" {
		t.Fatalf("rocm selection = %q", selected.Name)
	}
}

func TestResolveActiveRuntime(t *testing.T) {
	dir := t.TempDir()
	mgr := &Manager{runtimesDir: dir}

	runtimeDir := filepath.Join(dir, "llama-b9174")
	if err := os.MkdirAll(runtimeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	binaryPath := filepath.Join(runtimeDir, runtimeBinaryName())
	if err := os.WriteFile(binaryPath, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, activeFileName), []byte("llama-b9174\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := mgr.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if got != binaryPath {
		t.Fatalf("Resolve() = %q, want %q", got, binaryPath)
	}
}

func TestResolveNamedRuntime(t *testing.T) {
	dir := t.TempDir()
	mgr := &Manager{runtimesDir: dir}

	runtimeDir := filepath.Join(dir, "turbo")
	if err := os.MkdirAll(runtimeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	binaryPath := filepath.Join(runtimeDir, runtimeBinaryName())
	if err := os.WriteFile(binaryPath, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	got, err := mgr.ResolveNamed("turbo")
	if err != nil {
		t.Fatal(err)
	}
	if got != binaryPath {
		t.Fatalf("ResolveNamed() = %q, want %q", got, binaryPath)
	}
}

func TestRuntimeBackendReadsMetadataAndDefaultsToCUDA(t *testing.T) {
	dir := t.TempDir()
	mgr := &Manager{runtimesDir: dir}

	runtimeDir := filepath.Join(dir, "llama-vulkan")
	if err := os.MkdirAll(runtimeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runtimeDir, "backend"), []byte("vulkan\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := mgr.RuntimeBackend("llama-vulkan"); got != BuildBackendVulkan {
		t.Fatalf("RuntimeBackend() = %q, want vulkan", got)
	}

	rocmDir := filepath.Join(dir, "llama-rocm")
	if err := os.MkdirAll(rocmDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rocmDir, "backend"), []byte("rocm\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := mgr.RuntimeBackend("llama-rocm"); got != BuildBackendROCm {
		t.Fatalf("RuntimeBackend() = %q, want rocm", got)
	}

	cpuDir := filepath.Join(dir, "llama-cpu")
	if err := os.MkdirAll(cpuDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if got := mgr.RuntimeBackend("llama-cpu"); got != BuildBackendCUDA {
		t.Fatalf("RuntimeBackend() without metadata = %q, want cuda default", got)
	}
}

func TestAddWritesBackendMetadata(t *testing.T) {
	dir := t.TempDir()
	mgr := &Manager{runtimesDir: dir}

	binaryPath := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(binaryPath, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	if err := mgr.Add("custom-runtime", binaryPath, BuildBackendVulkan); err != nil {
		t.Fatalf("Add() error: %v", err)
	}

	if got := mgr.RuntimeBackend("custom-runtime"); got != BuildBackendVulkan {
		t.Fatalf("RuntimeBackend() = %q, want vulkan", got)
	}

	data, err := os.ReadFile(filepath.Join(dir, "custom-runtime", "backend"))
	if err != nil {
		t.Fatalf("read backend metadata: %v", err)
	}
	if strings.TrimSpace(string(data)) != "vulkan" {
		t.Fatalf("backend metadata = %q, want vulkan", strings.TrimSpace(string(data)))
	}
}

func TestResolveNoActiveRuntime(t *testing.T) {
	dir := t.TempDir()
	mgr := &Manager{runtimesDir: dir}

	_, err := mgr.Resolve()
	if err == nil {
		t.Fatal("expected error when no active runtime exists")
	}
}

func TestSelectAssetRejectsSYCLAndVulkanOnNvidia(t *testing.T) {
	platform := Platform{OS: "linux", Arch: "amd64", CUDA: "available"}
	assets := []ReleaseAsset{
		{Name: "llama-b9219-bin-ubuntu-sycl-fp16-x64.tar.gz"},
		{Name: "llama-b9219-bin-ubuntu-vulkan-x64.tar.gz"},
		{Name: "llama-b9219-bin-ubuntu-x64.tar.gz"},
	}
	selected, err := SelectAsset(assets, platform)
	if err != nil {
		t.Fatalf("SelectAsset failed: %v", err)
	}
	if strings.Contains(selected.Name, "sycl") || strings.Contains(selected.Name, "vulkan") {
		t.Errorf("expected SYCL/Vulkan to be rejected on NVIDIA, got %q", selected.Name)
	}
	if selected.Name != "llama-b9219-bin-ubuntu-x64.tar.gz" {
		t.Errorf("expected generic ubuntu-x64 to win, got %q", selected.Name)
	}
}

func TestSelectAssetPrefersCudaWhenAvailable(t *testing.T) {
	platform := Platform{OS: "linux", Arch: "amd64", CUDA: "available"}
	assets := []ReleaseAsset{
		{Name: "llama-b9219-bin-ubuntu-sycl-fp16-x64.tar.gz"},
		{Name: "llama-b9219-bin-ubuntu-cuda-cu12.4-x64.tar.gz"},
		{Name: "llama-b9219-bin-ubuntu-x64.tar.gz"},
	}
	selected, err := SelectAsset(assets, platform)
	if err != nil {
		t.Fatalf("SelectAsset failed: %v", err)
	}
	if !strings.Contains(selected.Name, "cuda") {
		t.Errorf("expected CUDA asset, got %q", selected.Name)
	}
}

func TestHomeRuntimesDirEmptyWhenHOMEUnset(t *testing.T) {
	t.Setenv("HOME", "")
	// Reset the once so the warning would fire if HOME is empty — but
	// what we care about is that the function returns "" cleanly.
	homeUnsetWarnOnce = sync.Once{}
	var sink bytes.Buffer
	stderrWriter = &sink
	t.Cleanup(func() { stderrWriter = os.Stderr })

	if got := homeRuntimesDir(); got != "" {
		t.Errorf("homeRuntimesDir() with empty HOME = %q, want \"\"", got)
	}
	if !strings.Contains(sink.String(), "HOME not set") {
		t.Errorf("expected stderr warning, got %q", sink.String())
	}
}

func TestDetectBuildBackendCUDA(t *testing.T) {
	tmpDir := t.TempDir()
	nvcc := filepath.Join(tmpDir, "nvcc")
	if err := os.WriteFile(nvcc, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldLookPath := execLookPath
	oldFileStat := fileStat
	t.Cleanup(func() {
		execLookPath = oldLookPath
		fileStat = oldFileStat
	})

	execLookPath = func(file string) (string, error) {
		if file == "nvcc" {
			return nvcc, nil
		}
		return "", os.ErrNotExist
	}
	fileStat = func(string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	}

	got, err := detectBuildBackend()
	if err != nil {
		t.Fatal(err)
	}
	if got.backend != BuildBackendCUDA {
		t.Fatalf("backend = %q, want cuda", got.backend)
	}
	if got.label != "CUDA (nvcc found)" {
		t.Fatalf("label = %q", got.label)
	}
}

func TestDetectBuildBackendPrefersCUDAOverROCm(t *testing.T) {
	tmpDir := t.TempDir()
	nvcc := filepath.Join(tmpDir, "nvcc")
	hipcc := filepath.Join(tmpDir, "hipcc")
	for _, path := range []string{nvcc, hipcc} {
		if err := os.WriteFile(path, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	oldLookPath := execLookPath
	oldFileStat := fileStat
	t.Cleanup(func() {
		execLookPath = oldLookPath
		fileStat = oldFileStat
	})

	execLookPath = func(file string) (string, error) {
		switch file {
		case "nvcc":
			return nvcc, nil
		case "hipcc":
			return hipcc, nil
		default:
			return "", os.ErrNotExist
		}
	}
	fileStat = func(string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	}

	got, err := detectBuildBackend()
	if err != nil {
		t.Fatal(err)
	}
	if got.backend != BuildBackendCUDA {
		t.Fatalf("backend = %q, want cuda", got.backend)
	}
}

func TestDetectBuildBackendROCm(t *testing.T) {
	tmpDir := t.TempDir()
	hipcc := filepath.Join(tmpDir, "hipcc")
	if err := os.WriteFile(hipcc, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldLookPath := execLookPath
	oldFileStat := fileStat
	t.Cleanup(func() {
		execLookPath = oldLookPath
		fileStat = oldFileStat
	})

	execLookPath = func(file string) (string, error) {
		switch file {
		case "nvcc":
			return "", os.ErrNotExist
		case "hipcc":
			return hipcc, nil
		default:
			return "", os.ErrNotExist
		}
	}
	fileStat = func(string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	}

	got, err := detectBuildBackend()
	if err != nil {
		t.Fatal(err)
	}
	if got.backend != BuildBackendROCm {
		t.Fatalf("backend = %q, want rocm", got.backend)
	}
	if got.label != "ROCm (hipcc found)" {
		t.Fatalf("label = %q", got.label)
	}
}

func TestDetectBuildBackendVulkan(t *testing.T) {
	tmpDir := t.TempDir()
	glslc := filepath.Join(tmpDir, "glslc")
	if err := os.WriteFile(glslc, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldLookPath := execLookPath
	oldFileStat := fileStat
	t.Cleanup(func() {
		execLookPath = oldLookPath
		fileStat = oldFileStat
	})

	execLookPath = func(file string) (string, error) {
		switch file {
		case "nvcc", "pkg-config":
			return "", os.ErrNotExist
		case "glslc":
			return glslc, nil
		default:
			return "", os.ErrNotExist
		}
	}
	fileStat = func(path string) (os.FileInfo, error) {
		if path == "/usr/include/vulkan/vulkan.h" {
			return nil, nil
		}
		return nil, os.ErrNotExist
	}

	got, err := detectBuildBackend()
	if err != nil {
		t.Fatal(err)
	}
	if got.backend != BuildBackendVulkan {
		t.Fatalf("backend = %q, want vulkan", got.backend)
	}
	if got.label != "Vulkan (libvulkan + glslc found)" {
		t.Fatalf("label = %q", got.label)
	}
}

func TestDetectBuildBackendCPU(t *testing.T) {
	oldLookPath := execLookPath
	oldFileStat := fileStat
	t.Cleanup(func() {
		execLookPath = oldLookPath
		fileStat = oldFileStat
	})

	execLookPath = func(string) (string, error) {
		return "", os.ErrNotExist
	}
	fileStat = func(string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	}

	got, err := detectBuildBackend()
	if err != nil {
		t.Fatal(err)
	}
	if got.backend != BuildBackendCPU {
		t.Fatalf("backend = %q, want cpu", got.backend)
	}
	if got.label != "CPU only (no GPU toolkit found)" {
		t.Fatalf("label = %q", got.label)
	}
}

func TestResolveBuildBackendVulkanOverridesAuto(t *testing.T) {
	tmpDir := t.TempDir()
	nvcc := filepath.Join(tmpDir, "nvcc")
	glslc := filepath.Join(tmpDir, "glslc")
	for _, path := range []string{nvcc, glslc} {
		if err := os.WriteFile(path, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	oldLookPath := execLookPath
	oldFileStat := fileStat
	t.Cleanup(func() {
		execLookPath = oldLookPath
		fileStat = oldFileStat
	})

	execLookPath = func(file string) (string, error) {
		switch file {
		case "nvcc":
			return nvcc, nil
		case "glslc":
			return glslc, nil
		case "pkg-config":
			return "", os.ErrNotExist
		default:
			return "", os.ErrNotExist
		}
	}
	fileStat = func(path string) (os.FileInfo, error) {
		if path == "/usr/include/vulkan/vulkan.h" {
			return nil, nil
		}
		return nil, os.ErrNotExist
	}

	got, err := resolveBuildBackend("vulkan")
	if err != nil {
		t.Fatal(err)
	}
	if got.backend != BuildBackendVulkan {
		t.Fatalf("backend = %q, want vulkan", got.backend)
	}
}

func TestResolveBuildBackendROCmOverridesAuto(t *testing.T) {
	tmpDir := t.TempDir()
	hipcc := filepath.Join(tmpDir, "hipcc")
	if err := os.WriteFile(hipcc, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldLookPath := execLookPath
	oldFileStat := fileStat
	t.Cleanup(func() {
		execLookPath = oldLookPath
		fileStat = oldFileStat
	})

	execLookPath = func(file string) (string, error) {
		if file == "hipcc" {
			return hipcc, nil
		}
		return "", os.ErrNotExist
	}
	fileStat = func(string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	}

	got, err := resolveBuildBackend("rocm")
	if err != nil {
		t.Fatal(err)
	}
	if got.backend != BuildBackendROCm {
		t.Fatalf("backend = %q, want rocm", got.backend)
	}
}

func TestDefaultBuildRuntimeName(t *testing.T) {
	if got := defaultBuildRuntimeName(BuildBackendROCm, "https://example.com/ggml-org/llama.cpp.git", "main"); got != "llama-rocm" {
		t.Fatalf("rocm name = %q, want llama-rocm", got)
	}
	if got := defaultBuildRuntimeName(BuildBackendVulkan, "https://example.com/ggml-org/llama.cpp.git", "main"); got != "llama-vulkan" {
		t.Fatalf("vulkan name = %q, want llama-vulkan", got)
	}
	if got := defaultBuildRuntimeName(BuildBackendCPU, "https://example.com/ggml-org/llama.cpp.git", "main"); got != "llama-cpu" {
		t.Fatalf("cpu name = %q, want llama-cpu", got)
	}
}

func TestBuildCMakeArgsIncludesVulkan(t *testing.T) {
	args := buildCMakeArgs(BuildBackendVulkan)
	found := false
	for _, arg := range args {
		if arg == "-DGGML_VULKAN=ON" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected Vulkan cmake flag in %v", args)
	}
}

func TestBuildCMakeArgsIncludesROCm(t *testing.T) {
	tmpDir := t.TempDir()
	rocminfo := filepath.Join(tmpDir, "rocminfo")
	if err := os.WriteFile(rocminfo, []byte("#!/bin/sh\ncat <<'EOF'\nName: gfx90a\nName: gfx1100\nEOF\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	oldLookPath := execLookPath
	t.Cleanup(func() {
		execLookPath = oldLookPath
	})

	execLookPath = func(file string) (string, error) {
		if file == "rocminfo" {
			return rocminfo, nil
		}
		return "", os.ErrNotExist
	}

	args := buildCMakeArgs(BuildBackendROCm)
	hasHIP := false
	hasTargets := false
	for _, arg := range args {
		if arg == "-DGGML_HIP=ON" {
			hasHIP = true
		}
		if strings.HasPrefix(arg, "-DGPU_TARGETS=") && strings.Contains(arg, "gfx90a") && strings.Contains(arg, "gfx1100") {
			hasTargets = true
		}
	}
	if !hasHIP {
		t.Fatalf("expected ROCm cmake flag in %v", args)
	}
	if !hasTargets {
		t.Fatalf("expected ROCm GPU targets in %v", args)
	}
}

func TestDetectROCmGPUTargetsIgnoresGenericSuffix(t *testing.T) {
	tmpDir := t.TempDir()
	rocminfo := filepath.Join(tmpDir, "rocminfo")
	if err := os.WriteFile(rocminfo, []byte(`#!/bin/sh
printf '%s\n' \
  'Name: gfx1201' \
  'Name: gfx12-generic'
`), 0o755); err != nil {
		t.Fatal(err)
	}

	oldLookPath := execLookPath
	t.Cleanup(func() {
		execLookPath = oldLookPath
	})

	execLookPath = func(file string) (string, error) {
		if file == "rocminfo" {
			return rocminfo, nil
		}
		return "", os.ErrNotExist
	}

	got := detectROCmGPUTargets()
	if got != "gfx1201" {
		t.Fatalf("detectROCmGPUTargets() = %q, want gfx1201", got)
	}
}

func TestStripTopDir(t *testing.T) {
	cases := map[string]string{
		"llama-b9275/llama-server":  "llama-server",
		"llama-b9275/lib/libfoo.so": "lib/libfoo.so",
		"foo":                       "foo",
		"/llama-b9275/llama-server": "llama-server",
		"":                          "",
	}
	for in, want := range cases {
		if got := stripTopDir(in); got != want {
			t.Errorf("stripTopDir(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestExtractRuntimeFromTarGzFlattensAndExtractsAll(t *testing.T) {
	dir := t.TempDir()
	archivePath := filepath.Join(dir, "llama.tar.gz")

	// Build a tarball wrapping every file under llama-b9275/.
	files := map[string][]byte{
		"llama-b9275/llama-server":         []byte("BINARY"),
		"llama-b9275/libllama.so":          []byte("LIB1"),
		"llama-b9275/libllama-common.so.0": []byte("LIB2"),
		"llama-b9275/README.md":            []byte("readme"),
	}
	writeTarGz(t, archivePath, files)

	outDir := filepath.Join(dir, "out")
	if err := extractRuntime(archivePath, outDir); err != nil {
		t.Fatalf("extractRuntime: %v", err)
	}
	for _, name := range []string{"llama-server", "libllama.so", "libllama-common.so.0", "README.md"} {
		path := filepath.Join(outDir, name)
		if _, err := os.Stat(path); err != nil {
			t.Errorf("expected %s in output dir: %v", name, err)
		}
	}
	data, err := os.ReadFile(filepath.Join(outDir, "llama-server"))
	if err != nil || string(data) != "BINARY" {
		t.Errorf("llama-server content mismatch: %v / %q", err, data)
	}
}

func TestExtractRuntimeFromZipFlattensAndExtractsAll(t *testing.T) {
	dir := t.TempDir()
	archivePath := filepath.Join(dir, "llama.zip")
	files := map[string][]byte{
		"llama-b9275/llama-server": []byte("BINARY"),
		"llama-b9275/libfoo.so":    []byte("LIB"),
	}
	writeZip(t, archivePath, files)

	outDir := filepath.Join(dir, "out")
	if err := extractRuntime(archivePath, outDir); err != nil {
		t.Fatalf("extractRuntime: %v", err)
	}
	for _, name := range []string{"llama-server", "libfoo.so"} {
		if _, err := os.Stat(filepath.Join(outDir, name)); err != nil {
			t.Errorf("expected %s in output dir: %v", name, err)
		}
	}
}

func TestDeriveBuildRuntimeName(t *testing.T) {
	cases := []struct {
		repo, branch, want string
	}{
		{"https://github.com/ggml-org/llama.cpp.git", "", "llama-build"},
		{"https://github.com/ggml-org/llama.cpp.git", "master", "llama-build-master"},
		{"https://github.com/ikawrakow/ik_llama.cpp.git", "", "ik_llama.cpp"},
		{"https://github.com/turboquant/turboquant.git", "v2", "turboquant-v2"},
		{"https://github.com/ggerganov/llama.cpp.git", "experiment/foo", "llama-build-experiment-foo"},
	}
	for _, c := range cases {
		if got := deriveBuildRuntimeName(c.repo, c.branch); got != c.want {
			t.Errorf("deriveBuildRuntimeName(%q, %q) = %q, want %q", c.repo, c.branch, got, c.want)
		}
	}
}

func TestBuildPrereqsAggregateMissingTools(t *testing.T) {
	oldLookPath := execLookPath
	t.Cleanup(func() {
		execLookPath = oldLookPath
	})

	execLookPath = func(string) (string, error) {
		return "", os.ErrNotExist
	}

	_, err := checkBuildTools(BuildBackendCUDA)
	if err == nil {
		t.Fatal("expected prerequisite check to fail when build tools are missing")
	}
	want := "sudo apt install git cmake build-essential nvidia-cuda-toolkit"
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("prerequisite error = %v, want command %q", err, want)
	}
}

func TestAutoBuildPrereqsAggregateMissingTools(t *testing.T) {
	oldLookPath := execLookPath
	t.Cleanup(func() {
		execLookPath = oldLookPath
	})

	execLookPath = func(string) (string, error) {
		return "", os.ErrNotExist
	}

	_, _, err := checkAutoBuildPrereqs(BuildBackendROCm)
	if err == nil {
		t.Fatal("expected auto-build prerequisite check to fail when build tools are missing")
	}
	want := "sudo apt install git cmake build-essential rocm-hip-sdk"
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("prerequisite error = %v, want command %q", err, want)
	}
}

func TestInstallTriggersAutoBuildWhenCudaAssetMissing(t *testing.T) {
	dir := t.TempDir()
	mgr := &Manager{runtimesDir: dir}

	oldDetectPlatform := detectPlatformFn
	oldFetchLatest := fetchLatestReleaseFn
	oldFetchRelease := fetchReleaseFn
	oldAutoBuild := autoBuildRuntimeFn
	t.Cleanup(func() {
		detectPlatformFn = oldDetectPlatform
		fetchLatestReleaseFn = oldFetchLatest
		fetchReleaseFn = oldFetchRelease
		autoBuildRuntimeFn = oldAutoBuild
	})

	detectPlatformFn = func() Platform {
		return Platform{OS: "linux", Arch: "amd64", CUDA: "available"}
	}
	fetchLatestReleaseFn = func() (*Release, error) {
		return &Release{
			TagName: "b123",
			Assets: []ReleaseAsset{
				{Name: "llama-b123-bin-ubuntu-x64.zip", DownloadURL: "https://example.com/cpu.zip", Size: 111},
			},
		}, nil
	}
	fetchReleaseFn = func(string) (*Release, error) {
		t.Fatal("FetchRelease should not be called when version is empty")
		return nil, nil
	}

	called := false
	autoBuildRuntimeFn = func(m *Manager, name, ref string, backend BuildBackend) (*RuntimeInfo, error) {
		called = true
		if name != "llama-b123" {
			t.Fatalf("AutoBuild name = %q, want llama-b123", name)
		}
		if ref != "b123" {
			t.Fatalf("AutoBuild ref = %q, want b123", ref)
		}
		if backend != BuildBackendCUDA {
			t.Fatalf("AutoBuild backend = %q, want cuda", backend)
		}
		return &RuntimeInfo{
			Name:    name,
			Path:    filepath.Join(m.runtimesDir, name, runtimeBinaryName()),
			Version: "b123",
			Source:  "build",
			Active:  true,
		}, nil
	}

	info, err := mgr.Install("", false)
	if err != nil {
		t.Fatalf("Install() error: %v", err)
	}
	if !called {
		t.Fatal("expected Install to trigger AutoBuild")
	}
	if info.Source != "build" {
		t.Fatalf("Install() source = %q, want build", info.Source)
	}
}

// writeTarGz creates a minimal tar.gz containing the given files.
func writeTarGz(t *testing.T, path string, files map[string][]byte) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	for name, data := range files {
		hdr := &tar.Header{
			Name: name,
			Mode: 0o644,
			Size: int64(len(data)),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeZip(t *testing.T, path string, files map[string][]byte) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for name, data := range files {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}
