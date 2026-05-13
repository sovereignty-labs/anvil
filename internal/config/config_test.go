package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Bind != "0.0.0.0:11434" {
		t.Errorf("expected bind 0.0.0.0:11434, got %s", cfg.Bind)
	}

	if cfg.ModelDir == "" {
		t.Error("expected non-empty default model_dir")
	}

	if cfg.Defaults["flash-attn"] != "on" {
		t.Error("expected flash-attn default to be 'on'")
	}
}

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	yaml := `
model_dir: /mnt/models/gguf
bind: 0.0.0.0:9999

remotes:
  gpu-host:
    url: http://gpu-host.example.internal:11434

autoload:
  - model: gemma-4-26B-A4B-Q3_K_XL.gguf
    gpu: 0
    runtime: turboquant
    flags:
      ctx-size: 131072
      parallel: 8
      flash-attn: "on"
      cache-type-k: q8_0
      cache-type-v: turbo3

  - model: qwen3.6-27B-IQ4_XS.gguf
    device: cpu
    flags:
      ctx-size: 65536
      threads: 12

aliases:
  advisor: qwen3.6-27B-IQ4_XS
  builder: gemma-4-26B-A4B-Q3_K_XL

defaults:
  flash-attn: "on"
  no-warmup: true
  jinja: true
`
	if err := os.WriteFile(cfgPath, []byte(yaml), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(cfgPath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if cfg.ModelDir != "/mnt/models/gguf" {
		t.Errorf("model_dir: got %s, want /mnt/models/gguf", cfg.ModelDir)
	}

	if cfg.Bind != "0.0.0.0:9999" {
		t.Errorf("bind: got %s, want 0.0.0.0:9999", cfg.Bind)
	}

	if len(cfg.Remotes) != 1 {
		t.Fatalf("expected 1 remote, got %d", len(cfg.Remotes))
	}
	if cfg.Remotes["gpu-host"].URL != "http://gpu-host.example.internal:11434" {
		t.Errorf("remote gpu-host url: got %s", cfg.Remotes["gpu-host"].URL)
	}

	if len(cfg.Autoload) != 2 {
		t.Fatalf("expected 2 autoload entries, got %d", len(cfg.Autoload))
	}

	entry0 := cfg.Autoload[0]
	if entry0.Model != "gemma-4-26B-A4B-Q3_K_XL.gguf" {
		t.Errorf("autoload[0].model: got %s", entry0.Model)
	}
	if entry0.GPU == nil || *entry0.GPU != 0 {
		t.Error("autoload[0].gpu should be 0")
	}
	if entry0.Runtime != "turboquant" {
		t.Errorf("autoload[0].runtime: got %s", entry0.Runtime)
	}

	entry1 := cfg.Autoload[1]
	if entry1.Device != "cpu" {
		t.Errorf("autoload[1].device: got %s", entry1.Device)
	}
	if entry1.GPU != nil {
		t.Error("autoload[1].gpu should be nil")
	}

	if len(cfg.Aliases) != 2 {
		t.Fatalf("expected 2 aliases, got %d", len(cfg.Aliases))
	}
}

func TestMergedFlags(t *testing.T) {
	cfg := &Config{
		Defaults: map[string]interface{}{
			"flash-attn": "on",
			"no-warmup":  true,
			"jinja":      true,
		},
	}

	entry := AutoloadEntry{
		Flags: map[string]interface{}{
			"ctx-size":     131072,
			"flash-attn":   "off", // overrides default
			"cache-type-k": "q8_0",
		},
	}

	merged := cfg.MergedFlags(entry)

	if merged["flash-attn"] != "off" {
		t.Error("entry flag should override default")
	}
	if merged["no-warmup"] != true {
		t.Error("default should be preserved when not overridden")
	}
	if merged["ctx-size"] != 131072 {
		t.Error("entry flag should be present")
	}
	if merged["cache-type-k"] != "q8_0" {
		t.Error("entry flag should be present")
	}
}

func TestResolveAlias(t *testing.T) {
	cfg := &Config{
		Aliases: map[string]string{
			"advisor": "qwen3.6-27B-IQ4_XS",
			"builder": "gemma-4-26B-A4B-Q3_K_XL",
		},
	}

	if got := cfg.ResolveAlias("advisor"); got != "qwen3.6-27B-IQ4_XS" {
		t.Errorf("expected alias resolution, got %s", got)
	}

	if got := cfg.ResolveAlias("unknown-model"); got != "unknown-model" {
		t.Errorf("expected passthrough for unknown alias, got %s", got)
	}
}

func TestModelPath(t *testing.T) {
	cfg := &Config{ModelDir: "/mnt/models/gguf"}

	// Relative filename
	if got := cfg.ModelPath("model.gguf"); got != "/mnt/models/gguf/model.gguf" {
		t.Errorf("expected joined path, got %s", got)
	}

	// Absolute filename passes through
	if got := cfg.ModelPath("/tmp/model.gguf"); got != "/tmp/model.gguf" {
		t.Errorf("expected absolute passthrough, got %s", got)
	}
}

func TestLoadMissingFile(t *testing.T) {
	_, err := Load("/nonexistent/config.yaml")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestLoadInvalidYAML(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "bad.yaml")
	os.WriteFile(cfgPath, []byte("{{not yaml}}"), 0644)

	_, err := Load(cfgPath)
	if err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestFindConfigNoFile(t *testing.T) {
	// With no config files at standard locations, should return empty
	// This test assumes the test environment doesn't have nollama configs
	// which is a reasonable assumption
	path := FindConfig()
	// We can't assert empty because the test machine might have one,
	// but we can assert it doesn't panic
	_ = path
}
