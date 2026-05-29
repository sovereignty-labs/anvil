package runtime

import (
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	stdruntime "runtime"
	"strings"
)

const defaultBuildRepo = "https://github.com/ggml-org/llama.cpp.git"

// buildTools holds the resolved paths of build prerequisites. Empty strings
// mean "not on PATH".
type buildTools struct {
	git   string
	cmake string
	make  string
	ninja string
	nvcc  string
	hipcc string
}

func collectBuildTools() buildTools {
	return buildTools{
		git:   lookPathOrEmpty("git"),
		cmake: lookPathOrEmpty("cmake"),
		make:  lookPathOrEmpty("make"),
		ninja: lookPathOrEmpty("ninja"),
		nvcc:  lookPathOrEmpty("nvcc"),
		hipcc: lookPathOrEmpty("hipcc"),
	}
}

func appendMissingPackage(missing []string, seen map[string]struct{}, pkg string) []string {
	if _, ok := seen[pkg]; ok {
		return missing
	}
	seen[pkg] = struct{}{}
	return append(missing, pkg)
}

func buildPrereqError(missing []string) error {
	return fmt.Errorf("Error: missing build dependencies. Install with:\n  sudo apt install %s\n\nOn other distros, install the equivalents of cmake, gcc/build-essential, make or ninja, and cuda-toolkit or rocm-hip-sdk as needed.", strings.Join(missing, " "))
}

// checkBuildTools resolves the required toolchain on PATH and reports all
// missing dependencies in one error.
func checkBuildTools(backend BuildBackend) (buildTools, error) {
	t := collectBuildTools()
	seen := make(map[string]struct{}, 4)
	missing := make([]string, 0, 4)

	if t.git == "" {
		missing = appendMissingPackage(missing, seen, "git")
	}
	if t.cmake == "" {
		missing = appendMissingPackage(missing, seen, "cmake")
	}
	if findCXXCompiler() == "" {
		missing = appendMissingPackage(missing, seen, "build-essential")
	}
	if t.make == "" && t.ninja == "" {
		missing = appendMissingPackage(missing, seen, "build-essential")
	}

	switch backend {
	case BuildBackendCUDA:
		if t.nvcc == "" {
			missing = appendMissingPackage(missing, seen, "nvidia-cuda-toolkit")
		}
	case BuildBackendROCm:
		if t.hipcc == "" {
			missing = appendMissingPackage(missing, seen, "rocm-hip-sdk")
		}
	}

	if len(missing) > 0 {
		return t, buildPrereqError(missing)
	}
	return t, nil
}

func lookPathOrEmpty(name string) string {
	p, err := execLookPath(name)
	if err != nil {
		return ""
	}
	return p
}

// autoBuildBackendForPlatform picks the backend that should be built from
// source for the detected platform.
func autoBuildBackendForPlatform(platform Platform) BuildBackend {
	switch {
	case platform.CUDA != "":
		return BuildBackendCUDA
	case platform.ROCm != "":
		return BuildBackendROCm
	default:
		return BuildBackendCPU
	}
}

func backendDisplayName(backend BuildBackend) string {
	switch backend {
	case BuildBackendCUDA:
		return "CUDA"
	case BuildBackendROCm:
		return "ROCm"
	case BuildBackendVulkan:
		return "Vulkan"
	case BuildBackendCPU:
		return "CPU"
	default:
		return strings.ToUpper(string(backend))
	}
}

func findCXXCompiler() string {
	for _, name := range []string{"c++", "g++", "clang++"} {
		if path := lookPathOrEmpty(name); path != "" {
			return path
		}
	}
	return ""
}

func findCUDAToolkit() (string, string) {
	if nvcc := lookPathOrEmpty("nvcc"); nvcc != "" {
		return nvcc, filepath.Dir(filepath.Dir(nvcc))
	}

	candidates := []string{"/usr/local/cuda/bin/nvcc"}
	if matches, err := filepath.Glob("/usr/local/cuda-*/bin/nvcc"); err == nil {
		candidates = append(candidates, matches...)
	}
	for _, nvcc := range candidates {
		if info, err := os.Stat(nvcc); err == nil && !info.IsDir() {
			return nvcc, filepath.Dir(filepath.Dir(nvcc))
		}
	}
	return "", ""
}

func checkAutoBuildPrereqs(backend BuildBackend) (string, string, error) {
	t := collectBuildTools()
	seen := make(map[string]struct{}, 4)
	missing := make([]string, 0, 4)
	nvccPath := ""
	toolkitRoot := ""
	if t.git == "" {
		missing = appendMissingPackage(missing, seen, "git")
	}
	if t.cmake == "" {
		missing = appendMissingPackage(missing, seen, "cmake")
	}
	if findCXXCompiler() == "" {
		missing = appendMissingPackage(missing, seen, "build-essential")
	}
	if t.make == "" && t.ninja == "" {
		missing = appendMissingPackage(missing, seen, "build-essential")
	}
	switch backend {
	case BuildBackendCUDA:
		if nvcc, root := findCUDAToolkit(); root != "" {
			nvccPath = nvcc
			toolkitRoot = root
		} else {
			missing = appendMissingPackage(missing, seen, "nvidia-cuda-toolkit")
		}
	case BuildBackendROCm:
		if t.hipcc == "" {
			missing = appendMissingPackage(missing, seen, "rocm-hip-sdk")
		}
	}

	if len(missing) > 0 {
		return "", "", buildPrereqError(missing)
	}
	return nvccPath, toolkitRoot, nil
}

func autoBuildCMakeArgs(backend BuildBackend, toolkitRoot string, nvccPath string) []string {
	args := []string{
		"-B", "build",
		"-DCMAKE_BUILD_TYPE=Release",
		"-DBUILD_SHARED_LIBS=ON",
		"-DLLAMA_BUILD_TESTS=OFF",
		"-DLLAMA_BUILD_EXAMPLES=OFF",
		"-DLLAMA_BUILD_SERVER=ON",
	}

	switch backend {
	case BuildBackendCUDA:
		args = append(args, "-DGGML_CUDA=ON")
		if toolkitRoot != "" {
			args = append(args, "-DCUDAToolkit_ROOT="+toolkitRoot)
		}
		if nvccPath != "" {
			args = append(args, "-DCMAKE_CUDA_COMPILER="+nvccPath)
		}
	case BuildBackendROCm:
		args = append(args, "-DGGML_HIP=ON", "-DGPU_TARGETS="+detectROCmGPUTargets())
	case BuildBackendVulkan:
		args = append(args, "-DGGML_VULKAN=ON")
	}

	return args
}

func runBuildCmdEnv(dir string, env []string, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if env != nil {
		cmd.Env = env
	}
	return cmd.Run()
}

func buildEnvWithCompiler(compiler string) []string {
	env := os.Environ()
	if compiler != "" {
		env = append(env, "CXX="+compiler)
	}
	return env
}

func buildJobs() string {
	return fmt.Sprintf("%d", stdruntime.NumCPU())
}

// AutoBuild builds llama-server from source using the backend detected on the
// current machine.
func (m *Manager) AutoBuild(name string) error {
	backend := autoBuildBackendForPlatform(DetectPlatform())
	_, err := m.autoBuild(name, "", backend)
	return err
}

func (m *Manager) autoBuild(name, ref string, backend BuildBackend) (*RuntimeInfo, error) {
	if err := m.ensureDir(); err != nil {
		return nil, fmt.Errorf("prepare runtimes dir: %w", err)
	}

	if err := validateRuntimeName(name); err != nil {
		return nil, err
	}

	nvccPath, toolkitRoot, err := checkAutoBuildPrereqs(backend)
	if err != nil {
		return nil, err
	}

	srcDir, err := os.MkdirTemp("", "nollama-build-")
	if err != nil {
		return nil, fmt.Errorf("create build temp dir: %w", err)
	}
	defer os.RemoveAll(srcDir)

	repo := defaultBuildRepo
	cloneArgs := []string{"clone", "--depth", "1"}
	if b := strings.TrimSpace(ref); b != "" {
		cloneArgs = append(cloneArgs, "--branch", b, "--single-branch")
	}
	cloneArgs = append(cloneArgs, repo, srcDir)

	compiler := findCXXCompiler()

	fmt.Fprintf(os.Stderr, "\nCloning %s...\n", repo)
	if err := runBuildCmdEnv(srcDir, buildEnvWithCompiler(compiler), "git", cloneArgs...); err != nil {
		return nil, fmt.Errorf("git clone: %w", err)
	}

	cmakeArgs := autoBuildCMakeArgs(backend, toolkitRoot, nvccPath)
	fmt.Fprintf(os.Stderr, "\nConfiguring: cmake %s\n", strings.Join(cmakeArgs, " "))
	if err := runBuildCmdEnv(srcDir, buildEnvWithCompiler(compiler), "cmake", cmakeArgs...); err != nil {
		return nil, fmt.Errorf("cmake configure: %w", err)
	}

	jobs := buildJobs()
	fmt.Fprintf(os.Stderr, "\nBuilding (-j%s)...\n", jobs)
	if err := runBuildCmdEnv(srcDir, buildEnvWithCompiler(compiler), "cmake", "--build", "build", "--config", "Release", "-j", jobs, "--target", "llama-server"); err != nil {
		return nil, fmt.Errorf("cmake build: %w", err)
	}

	binarySrc, libs, err := findBuildArtifacts(filepath.Join(srcDir, "build"))
	if err != nil {
		return nil, err
	}

	finalDir, err := m.runtimeDir(name)
	if err != nil {
		return nil, err
	}
	workDir, err := os.MkdirTemp(m.runtimesDir, "."+name+".")
	if err != nil {
		return nil, fmt.Errorf("create temp runtime dir: %w", err)
	}
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.RemoveAll(workDir)
		}
	}()

	if err := copyFile(binarySrc, filepath.Join(workDir, runtimeBinaryName())); err != nil {
		return nil, err
	}
	for _, lib := range libs {
		if err := copyFile(lib, filepath.Join(workDir, filepath.Base(lib))); err != nil {
			return nil, err
		}
	}
	if err := chmodRuntimeArtifacts(workDir); err != nil {
		return nil, err
	}
	if err := writeSourceMarker(workDir, "build"); err != nil {
		return nil, err
	}
	if err := writeBackendMarker(workDir, backend); err != nil {
		return nil, err
	}

	if err := os.RemoveAll(finalDir); err != nil {
		return nil, fmt.Errorf("remove existing runtime %s: %w", finalDir, err)
	}
	if err := os.Rename(workDir, finalDir); err != nil {
		return nil, fmt.Errorf("finalize build runtime: %w", err)
	}
	cleanup = false

	if err := m.Use(name); err != nil {
		return nil, err
	}

	info := RuntimeInfo{
		Name:    name,
		Path:    filepath.Join(finalDir, runtimeBinaryName()),
		Version: "",
		Source:  "build",
		Active:  true,
	}
	fmt.Fprintf(os.Stderr, "Installed: %s\n", info.Path)
	fmt.Fprintf(os.Stderr, "Active runtime: %s ✓\n", info.Name)
	return &info, nil
}

// printBuildTools prints the resolved toolchain to stderr.
func printBuildTools(t buildTools) {
	fmt.Fprintln(os.Stderr, "Checking build tools...")
	row := func(name, path string) {
		mark := "✓"
		if path == "" {
			mark = "✗"
			path = "(not found)"
		}
		fmt.Fprintf(os.Stderr, "  %-7s %s %s\n", name+":", path, mark)
	}
	row("git", t.git)
	row("cmake", t.cmake)
	if t.ninja != "" {
		row("ninja", t.ninja)
	} else {
		row("make", t.make)
	}
	if t.nvcc != "" {
		row("nvcc", t.nvcc)
	}
	if t.hipcc != "" {
		row("hipcc", t.hipcc)
	}
}

// runBuildCmd executes name+args in dir, streaming stdout/stderr to the
// parent stderr so the user sees compilation progress in real time.
func runBuildCmd(dir, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// findBuildArtifacts walks buildDir to locate the llama-server binary and
// every .so file produced by the build. Returns the binary path and a slice
// of shared-library paths.
func findBuildArtifacts(buildDir string) (string, []string, error) {
	binaryName := runtimeBinaryName()
	binaryPath := ""
	var libs []string

	err := filepath.WalkDir(buildDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		name := d.Name()
		if name == binaryName && binaryPath == "" {
			binaryPath = path
			return nil
		}
		// .so, .so.0, .so.1.2 — all match. Skip object files and other intermediates.
		if strings.Contains(name, ".so") && !strings.HasSuffix(name, ".o") {
			libs = append(libs, path)
		}
		return nil
	})
	if err != nil {
		return "", nil, fmt.Errorf("scan build artifacts: %w", err)
	}
	if binaryPath == "" {
		return "", nil, fmt.Errorf("llama-server binary not found under %s", buildDir)
	}
	return binaryPath, libs, nil
}

// deriveBuildRuntimeName picks a sensible runtime name from a repo URL and
// optional branch. Mainline llama.cpp gives "llama-build" (or
// "llama-build-<branch>"); forks use the repo name.
func deriveBuildRuntimeName(repo, branch string) string {
	branch = strings.TrimSpace(branch)
	repo = strings.TrimSpace(repo)
	repoLower := strings.ToLower(repo)

	base := "llama-build"
	if !strings.Contains(repoLower, "ggml-org/llama.cpp") && !strings.Contains(repoLower, "ggerganov/llama.cpp") {
		base = strings.TrimSuffix(filepath.Base(repo), ".git")
		if base == "" || base == "/" {
			base = "llama-build"
		}
	}
	if branch != "" {
		return base + "-" + sanitizeName(branch)
	}
	return base
}

// sanitizeName collapses anything that's not a runtime-name-safe rune into "-".
func sanitizeName(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			b.WriteRune(r)
		default:
			b.WriteRune('-')
		}
	}
	return b.String()
}
