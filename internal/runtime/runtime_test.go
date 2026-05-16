package runtime

import (
	"os"
	"path/filepath"
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
