package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// isolateModelDirs points HOME and XDG vars at a fresh tempdir so the
// resolver doesn't reach into the developer's real config / default model
// store. Returns the simulated default model dir.
func isolateModelDirs(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_DATA_HOME", "")
	t.Setenv("XDG_CONFIG_HOME", "")
	defaultDir := filepath.Join(home, ".local", "share", "nollama", "models")
	if err := os.MkdirAll(defaultDir, 0o755); err != nil {
		t.Fatal(err)
	}
	return defaultDir
}

// chdirTemp moves into an empty tempdir for the duration of the test so a
// stray .gguf in the developer's cwd can't satisfy the resolver.
func chdirTemp(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	orig, _ := os.Getwd()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(orig) })
}

func TestResolveModelPathAbsolute(t *testing.T) {
	isolateModelDirs(t)
	in := "/some/abs/path/model.gguf"
	got, err := resolveModelPath(in)
	if err != nil {
		t.Fatalf("expected absolute path to pass through, err: %v", err)
	}
	if got != in {
		t.Errorf("got %q, want %q", got, in)
	}
}

func TestResolveModelPathWithSeparator(t *testing.T) {
	isolateModelDirs(t)
	in := "subdir/model.gguf"
	got, err := resolveModelPath(in)
	if err != nil {
		t.Fatalf("expected separator path to pass through, err: %v", err)
	}
	if got != in {
		t.Errorf("got %q, want %q", got, in)
	}
}

func TestResolveModelPathBareFilenameInDefaultDir(t *testing.T) {
	defaultDir := isolateModelDirs(t)
	chdirTemp(t)
	target := filepath.Join(defaultDir, "model-Q4_K_M.gguf")
	if err := os.WriteFile(target, []byte("g"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := resolveModelPath("model-Q4_K_M.gguf")
	if err != nil {
		t.Fatalf("resolveModelPath: %v", err)
	}
	if got != target {
		t.Errorf("got %q, want %q", got, target)
	}
}

func TestResolveModelPathAutoGGUFSuffix(t *testing.T) {
	defaultDir := isolateModelDirs(t)
	chdirTemp(t)
	target := filepath.Join(defaultDir, "google_gemma-4-E2B-it-Q4_K_M.gguf")
	if err := os.WriteFile(target, []byte("g"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := resolveModelPath("google_gemma-4-E2B-it-Q4_K_M")
	if err != nil {
		t.Fatalf("resolveModelPath without .gguf: %v", err)
	}
	if got != target {
		t.Errorf("got %q, want %q", got, target)
	}
}

func TestResolveModelPathNotFoundListsSearched(t *testing.T) {
	defaultDir := isolateModelDirs(t)
	chdirTemp(t)

	_, err := resolveModelPath("nonexistent.gguf")
	if err == nil {
		t.Fatal("expected error when model not found")
	}
	msg := err.Error()
	if !strings.Contains(msg, "model not found") {
		t.Errorf("error missing 'model not found': %q", msg)
	}
	if !strings.Contains(msg, defaultDir) {
		t.Errorf("error should list default dir %q, got %q", defaultDir, msg)
	}
	if !strings.Contains(msg, "nollama pull") {
		t.Errorf("error should hint about pull, got %q", msg)
	}
}

func TestResolveModelPathCurrentDirWinsOverDefault(t *testing.T) {
	defaultDir := isolateModelDirs(t)
	cwd := t.TempDir()
	orig, _ := os.Getwd()
	if err := os.Chdir(cwd); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(orig) })

	cwdTarget := filepath.Join(cwd, "model.gguf")
	defaultTarget := filepath.Join(defaultDir, "model.gguf")
	if err := os.WriteFile(cwdTarget, []byte("cwd"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(defaultTarget, []byte("default"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := resolveModelPath("model.gguf")
	if err != nil {
		t.Fatalf("resolveModelPath: %v", err)
	}
	if got != cwdTarget {
		t.Errorf("got %q, want %q (cwd should win)", got, cwdTarget)
	}
}
