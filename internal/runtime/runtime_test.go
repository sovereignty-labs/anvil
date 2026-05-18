package runtime

import (
	"bytes"
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
