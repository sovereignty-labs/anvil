package runtime

import (
	"bufio"
	"bytes"
	"fmt"
	"regexp"
	"strings"
)

// BuildBackend selects the llama.cpp GPU backend used for `runtime build`.
type BuildBackend string

const (
	BuildBackendAuto   BuildBackend = ""
	BuildBackendCUDA   BuildBackend = "cuda"
	BuildBackendROCm   BuildBackend = "rocm"
	BuildBackendVulkan BuildBackend = "vulkan"
	BuildBackendCPU    BuildBackend = "cpu"
)

type backendSelection struct {
	backend BuildBackend
	label   string
}

func resolveBuildBackend(requested string) (backendSelection, error) {
	switch strings.ToLower(strings.TrimSpace(requested)) {
	case "", "auto":
		return detectBuildBackend()
	case string(BuildBackendCUDA):
		return backendSelection{backend: BuildBackendCUDA, label: "CUDA (forced by --backend)"}, nil
	case string(BuildBackendROCm):
		return backendSelection{backend: BuildBackendROCm, label: "ROCm (forced by --backend)"}, nil
	case string(BuildBackendVulkan):
		if err := ensureVulkanBuildPrereqs(); err != nil {
			return backendSelection{}, err
		}
		return backendSelection{backend: BuildBackendVulkan, label: "Vulkan (forced by --backend)"}, nil
	case string(BuildBackendCPU):
		return backendSelection{backend: BuildBackendCPU, label: "CPU only (forced by --backend)"}, nil
	default:
		return backendSelection{}, fmt.Errorf("invalid backend %q (expected cuda, rocm, vulkan, cpu)", requested)
	}
}

func detectBuildBackend() (backendSelection, error) {
	if lookPathOrEmpty("nvcc") != "" {
		return backendSelection{backend: BuildBackendCUDA, label: "CUDA (nvcc found)"}, nil
	}

	if lookPathOrEmpty("hipcc") != "" {
		return backendSelection{backend: BuildBackendROCm, label: "ROCm (hipcc found)"}, nil
	}

	if hasPkgConfigVulkan() {
		return backendSelection{backend: BuildBackendVulkan, label: "Vulkan (pkg-config vulkan found)"}, nil
	}

	if hasVulkanHeaders() && lookPathOrEmpty("glslc") != "" {
		return backendSelection{backend: BuildBackendVulkan, label: "Vulkan (libvulkan + glslc found)"}, nil
	}

	return backendSelection{backend: BuildBackendCPU, label: "CPU only (no GPU toolkit found)"}, nil
}

func hasPkgConfigVulkan() bool {
	pkgConfig := lookPathOrEmpty("pkg-config")
	if pkgConfig == "" {
		return false
	}
	cmd := execCommand(pkgConfig, "--exists", "vulkan")
	return cmd.Run() == nil
}

func hasVulkanHeaders() bool {
	_, err := fileStat("/usr/include/vulkan/vulkan.h")
	return err == nil
}

func ensureVulkanBuildPrereqs() error {
	if (hasPkgConfigVulkan() || hasVulkanHeaders()) && lookPathOrEmpty("glslc") != "" {
		return nil
	}

	return fmt.Errorf(`Vulkan build requires: libvulkan-dev, glslc, spirv-headers
Install on Ubuntu/Debian: sudo apt install -y libvulkan-dev glslc spirv-headers
Install on Fedora: sudo dnf install -y vulkan-devel glslc spirv-headers-devel`)
}

func ensureROCmBuildPrereqs() error {
	if lookPathOrEmpty("hipcc") != "" {
		return nil
	}

	return fmt.Errorf(`ROCm build requires hipcc on PATH
Install the ROCm HIP toolchain so hipcc is available to cmake`)
}

func buildCMakeArgs(backend BuildBackend) []string {
	args := []string{
		"-B", "build",
		"-DBUILD_SHARED_LIBS=ON",
		"-DLLAMA_BUILD_TESTS=OFF",
		"-DLLAMA_BUILD_EXAMPLES=OFF",
		"-DLLAMA_BUILD_SERVER=ON",
	}

	switch backend {
	case BuildBackendCUDA:
		args = append(args, "-DGGML_CUDA=ON")
	case BuildBackendROCm:
		args = append(args, "-DGGML_HIP=ON", "-DGPU_TARGETS="+detectROCmGPUTargets())
	case BuildBackendVulkan:
		args = append(args, "-DGGML_VULKAN=ON")
	}

	return args
}

func defaultBuildRuntimeName(backend BuildBackend, repo, branch string) string {
	switch backend {
	case BuildBackendROCm:
		return "llama-rocm"
	case BuildBackendVulkan:
		return "llama-vulkan"
	case BuildBackendCPU:
		return "llama-cpu"
	default:
		return deriveBuildRuntimeName(repo, branch)
	}
}

func (b BuildBackend) String() string {
	switch b {
	case BuildBackendCUDA:
		return string(BuildBackendCUDA)
	case BuildBackendROCm:
		return string(BuildBackendROCm)
	case BuildBackendVulkan:
		return string(BuildBackendVulkan)
	case BuildBackendCPU:
		return string(BuildBackendCPU)
	default:
		return "auto"
	}
}

var rocmTargetPattern = regexp.MustCompile(`(?i)gfx[0-9a-z]+`)

func detectROCmGPUTargets() string {
	rocminfo := lookPathOrEmpty("rocminfo")
	if rocminfo == "" {
		return "gfx1201"
	}

	out, err := execCommand(rocminfo).Output()
	if err != nil || len(out) == 0 {
		return "gfx1201"
	}

	seen := make(map[string]struct{})
	targets := make([]string, 0, 4)
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		line := scanner.Text()
		for _, match := range rocmTargetPattern.FindAllStringIndex(line, -1) {
			target := strings.ToLower(line[match[0]:match[1]])
			if rocmTargetIsGenericSuffix(line, match[1]) {
				continue
			}
			if _, ok := seen[target]; ok {
				continue
			}
			seen[target] = struct{}{}
			targets = append(targets, target)
		}
	}
	if len(targets) == 0 {
		return "gfx1201"
	}
	return strings.Join(targets, ";")
}

func rocmTargetIsGenericSuffix(line string, matchEnd int) bool {
	if matchEnd >= len(line) {
		return false
	}
	return strings.HasPrefix(strings.ToLower(line[matchEnd:]), "-generic")
}
