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

func TestDefaultBuildRuntimeName(t *testing.T) {
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

func TestSelectAssetWarnsWhenNoCUDABuild(t *testing.T) {
	var sink bytes.Buffer
	stderrWriter = &sink
	t.Cleanup(func() { stderrWriter = os.Stderr })

	platform := Platform{OS: "linux", Arch: "amd64", CUDA: "available"}
	assets := []ReleaseAsset{
		{Name: "llama-b9275-bin-ubuntu-x64.tar.gz"},
	}
	if _, err := SelectAsset(assets, platform); err != nil {
		t.Fatalf("SelectAsset: %v", err)
	}
	if !strings.Contains(sink.String(), "CUDA detected but no CUDA build") {
		t.Errorf("expected CUDA warning, got %q", sink.String())
	}
	if !strings.Contains(sink.String(), "nollama runtime build") {
		t.Errorf("expected hint to runtime build, got %q", sink.String())
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
