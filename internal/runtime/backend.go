package runtime

import (
	"fmt"
	"strings"
)

// BuildBackend selects the llama.cpp GPU backend used for `runtime build`.
type BuildBackend string

const (
	BuildBackendAuto   BuildBackend = ""
	BuildBackendCUDA   BuildBackend = "cuda"
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
		if lookPathOrEmpty("nvcc") == "" {
			return backendSelection{}, fmt.Errorf("cuda build requires nvcc on PATH")
		}
		return backendSelection{backend: BuildBackendCUDA, label: "CUDA (forced by --backend)"}, nil
	case string(BuildBackendVulkan):
		if err := ensureVulkanBuildPrereqs(); err != nil {
			return backendSelection{}, err
		}
		return backendSelection{backend: BuildBackendVulkan, label: "Vulkan (forced by --backend)"}, nil
	case string(BuildBackendCPU):
		return backendSelection{backend: BuildBackendCPU, label: "CPU only (forced by --backend)"}, nil
	default:
		return backendSelection{}, fmt.Errorf("invalid backend %q (expected cuda, vulkan, cpu)", requested)
	}
}

func detectBuildBackend() (backendSelection, error) {
	if lookPathOrEmpty("nvcc") != "" {
		return backendSelection{backend: BuildBackendCUDA, label: "CUDA (nvcc found)"}, nil
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
	case BuildBackendVulkan:
		args = append(args, "-DGGML_VULKAN=ON")
	}

	return args
}

func defaultBuildRuntimeName(backend BuildBackend, repo, branch string) string {
	switch backend {
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
	case BuildBackendVulkan:
		return string(BuildBackendVulkan)
	case BuildBackendCPU:
		return string(BuildBackendCPU)
	default:
		return "auto"
	}
}
