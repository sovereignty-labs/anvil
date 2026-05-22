package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// sandboxTokenHome points XDG_CONFIG_HOME and HOME at a tempdir so the
// resolver/writer don't touch the developer's real ~/.config/nollama/token.
func sandboxTokenHome(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("HOME", dir)
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("HF_TOKEN", "")
	return dir
}

func TestResolveHFTokenFromFileBeatsEnv(t *testing.T) {
	sandboxTokenHome(t)
	path := tokenFilePath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("file-token\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HF_TOKEN", "env-token")

	src := resolveHFTokenSource()
	if src.value != "file-token" {
		t.Errorf("expected file-token, got %q", src.value)
	}
	if !strings.HasSuffix(src.origin, "token") {
		t.Errorf("expected file-path origin, got %q", src.origin)
	}
}

func TestResolveHFTokenFallsBackToEnv(t *testing.T) {
	sandboxTokenHome(t)
	t.Setenv("HF_TOKEN", "env-token")

	src := resolveHFTokenSource()
	if src.value != "env-token" {
		t.Errorf("expected env-token, got %q", src.value)
	}
	if src.origin != "HF_TOKEN env" {
		t.Errorf("expected HF_TOKEN env origin, got %q", src.origin)
	}
}

func TestResolveHFTokenEmptyWhenNeitherSet(t *testing.T) {
	sandboxTokenHome(t)
	if got := resolveHFToken(); got != "" {
		t.Errorf("expected empty token, got %q", got)
	}
}

func TestTokenSetWritesFileWith0600(t *testing.T) {
	sandboxTokenHome(t)
	if err := runTokenSet(tokenSetCmd, []string{"hf_abc123"}); err != nil {
		t.Fatalf("runTokenSet: %v", err)
	}
	path := tokenFilePath()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat token file: %v", err)
	}
	if mode := info.Mode().Perm(); mode != 0o600 {
		t.Errorf("token file mode = %o, want 0600", mode)
	}
	data, _ := os.ReadFile(path)
	if strings.TrimSpace(string(data)) != "hf_abc123" {
		t.Errorf("token file content = %q", data)
	}
}

func TestTokenRoundTripSetShowRm(t *testing.T) {
	sandboxTokenHome(t)
	if err := runTokenSet(tokenSetCmd, []string{"hf_xyz"}); err != nil {
		t.Fatalf("set: %v", err)
	}
	if got := resolveHFToken(); got != "hf_xyz" {
		t.Errorf("after set, got %q", got)
	}
	if err := runTokenRM(tokenRMCmd, nil); err != nil {
		t.Fatalf("rm: %v", err)
	}
	if got := resolveHFToken(); got != "" {
		t.Errorf("after rm, expected empty, got %q", got)
	}
}

func TestMaskToken(t *testing.T) {
	cases := map[string]string{
		"hf_NCh-REDACTED-SECRET-sDSti": "hf_NCh...sDSti",
		"short":                                "*****",
		"":                                     "",
	}
	for in, want := range cases {
		if got := maskToken(in); got != want {
			t.Errorf("maskToken(%q) = %q, want %q", in, got, want)
		}
	}
}
