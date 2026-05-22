package runtime

import (
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
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
}

// checkBuildTools resolves the required toolchain on PATH. Returns an error
// when git or cmake is missing. nvcc is only required when platform.CUDA is
// available AND the caller wants a GPU build (we surface its presence but
// don't fail on absence — Build() decides whether to set -DGGML_CUDA=ON).
func checkBuildTools(platform Platform) (buildTools, error) {
	t := buildTools{
		git:   lookPathOrEmpty("git"),
		cmake: lookPathOrEmpty("cmake"),
		make:  lookPathOrEmpty("make"),
		ninja: lookPathOrEmpty("ninja"),
		nvcc:  lookPathOrEmpty("nvcc"),
	}
	var missing []string
	if t.git == "" {
		missing = append(missing, "git")
	}
	if t.cmake == "" {
		missing = append(missing, "cmake")
	}
	if t.make == "" && t.ninja == "" {
		missing = append(missing, "make or ninja")
	}
	if len(missing) > 0 {
		return t, fmt.Errorf("missing build tool(s): %s — install via your package manager (e.g. apt install build-essential cmake git)", strings.Join(missing, ", "))
	}
	if platform.CUDA != "" && t.nvcc == "" {
		fmt.Fprintln(stderrWriter, "Warning: CUDA detected but nvcc not on PATH; building CPU-only. Install the CUDA toolkit and re-run for GPU support.")
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
