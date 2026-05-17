// Package config handles nollama YAML configuration.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"gopkg.in/yaml.v3"
)

// Config is the top-level nollama configuration.
type Config struct {
	// ModelDir is where GGUFs live on this machine.
	// Default: ~/.local/share/nollama/models
	ModelDir string `yaml:"model_dir"`

	// LlamaServer overrides the llama-server binary path.
	// Normally managed by `nollama runtime` commands.
	LlamaServer string `yaml:"llama_server"`

	// Bind is the listen address for the unified API endpoint.
	// Default: 0.0.0.0:11434
	Bind string `yaml:"bind"`

	// Remotes are federated nollama nodes.
	Remotes map[string]Remote `yaml:"remotes"`

	// Autoload defines models to load on startup.
	Autoload []AutoloadEntry `yaml:"autoload"`

	// GPUs overrides auto-detected GPU info.
	GPUs map[int]GPUOverride `yaml:"gpus"`

	// Defaults are global flags applied to all loads (overridable per-model).
	Defaults map[string]interface{} `yaml:"defaults"`

	// Aliases map friendly names to model filenames for API routing.
	Aliases map[string]string `yaml:"aliases"`

	// Swap configures auto-swap behavior.
	Swap *SwapConfig `yaml:"swap"`

	// Management configures the management API.
	Management *ManagementConfig `yaml:"management"`

	// MCP configures the MCP server.
	MCP *MCPConfig `yaml:"mcp"`

	// RuntimesDir overrides the llama-server runtime directory.
	RuntimesDir string `yaml:"runtimes_dir"`
}

// Remote is a federated nollama node.
type Remote struct {
	URL string `yaml:"url"`
}

// AutoloadEntry defines a model to load on startup.
type AutoloadEntry struct {
	// Model is the GGUF filename (relative to model_dir).
	Model string `yaml:"model"`

	// GPU is the GPU index to load on. -1 or omitted = auto-select.
	GPU *int `yaml:"gpu"`

	// Device forces CPU inference when set to "cpu".
	Device string `yaml:"device"`

	// Runtime selects which llama-server runtime to use.
	Runtime string `yaml:"runtime"`

	// Profiles are named built-in flag bundles applied before explicit flags.
	Profiles []string `yaml:"profiles"`

	// Flags are llama-server flags for this model.
	Flags map[string]interface{} `yaml:"flags"`
}

// GPUOverride allows manual correction of auto-detected GPU info.
type GPUOverride struct {
	Name string `yaml:"name"`
	VRAM int    `yaml:"vram"` // MiB
}

// SwapConfig controls model auto-swap behavior.
type SwapConfig struct {
	Enabled     bool   `yaml:"enabled"`
	IdleTimeout string `yaml:"idle_timeout"` // e.g. "30m"
}

// ManagementConfig controls the management API.
type ManagementConfig struct {
	Bind string `yaml:"bind"` // Default: 127.0.0.1:11435
}

// MCPConfig controls the MCP server.
type MCPConfig struct {
	Enabled   bool   `yaml:"enabled"`
	Transport string `yaml:"transport"` // "stdio" or "sse"
	Bind      string `yaml:"bind"`      // Only used for SSE transport
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		ModelDir: defaultModelDir(),
		Bind:     "0.0.0.0:11434",
		Defaults: map[string]interface{}{
			"flash-attn": "on",
			"no-warmup":  true,
			"jinja":      true,
		},
	}
}

// Load reads a config file from the given path.
// Returns an error if the file cannot be read or parsed.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config %s: %w", path, err)
	}

	cfg := DefaultConfig()
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parsing config %s: %w", path, err)
	}

	// Resolve model_dir to absolute path
	if cfg.ModelDir != "" && !filepath.IsAbs(cfg.ModelDir) {
		abs, err := filepath.Abs(cfg.ModelDir)
		if err != nil {
			return nil, fmt.Errorf("resolving model_dir %s: %w", cfg.ModelDir, err)
		}
		cfg.ModelDir = abs
	}

	return cfg, nil
}

// FindConfig looks for a config file in standard locations.
// Returns the path if found, empty string if not.
func FindConfig() string {
	candidates := []string{}

	// XDG config
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		candidates = append(candidates, filepath.Join(xdg, "nollama", "config.yaml"))
	}

	// ~/.config/nollama/config.yaml
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates, filepath.Join(home, ".config", "nollama", "config.yaml"))
	}

	// /etc/nollama/config.yaml
	candidates = append(candidates, "/etc/nollama/config.yaml")

	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	return ""
}

// MergedFlags returns the autoload entry's flags merged on top of global defaults.
// Profile flags override defaults and explicit entry flags override profiles.
func (c *Config) MergedFlags(entry AutoloadEntry) map[string]interface{} {
	merged, _, err := c.MergedFlagsWithProfiles(entry)
	if err != nil {
		// Preserve the historical signature for callers that only need flags.
		// Profile lookup failures are handled by MergedFlagsWithProfiles users.
		return cloneFlags(c.Defaults)
	}
	return merged
}

// MergedFlagsWithProfiles merges defaults, profiles, and entry flags in order.
func (c *Config) MergedFlagsWithProfiles(entry AutoloadEntry) (map[string]interface{}, []ProfileRequires, error) {
	merged := make(map[string]interface{})

	// Start with global defaults
	for k, v := range c.Defaults {
		merged[k] = v
	}

	if len(entry.Profiles) > 0 {
		loaded := make([]Profile, 0, len(entry.Profiles))
		for _, name := range entry.Profiles {
			profile, err := LoadProfile(name)
			if err != nil {
				return nil, nil, err
			}
			loaded = append(loaded, profile)
		}
		profileMerge := MergeProfiles(loaded)
		for k, v := range profileMerge.Flags {
			merged[k] = v
		}

		// Explicit entry runtime takes precedence over profile requirements.
		reqs := append([]ProfileRequires(nil), profileMerge.Requires...)
		if entry.Runtime != "" {
			reqs = append(reqs, ProfileRequires{Runtime: entry.Runtime})
		}

		// Override with per-model flags
		for k, v := range entry.Flags {
			merged[k] = v
		}
		return merged, reqs, nil
	}

	// Override with per-model flags
	for k, v := range entry.Flags {
		merged[k] = v
	}

	reqs := []ProfileRequires{}
	if entry.Runtime != "" {
		reqs = append(reqs, ProfileRequires{Runtime: entry.Runtime})
	}
	return merged, reqs, nil
}

// ModelPath returns the full path to a model file.
func (c *Config) ModelPath(filename string) string {
	if filepath.IsAbs(filename) {
		return filename
	}
	return filepath.Join(c.ModelDir, filename)
}

// ResolveAlias returns the model filename for an alias, or the input if no alias exists.
func (c *Config) ResolveAlias(name string) string {
	if c.Aliases != nil {
		if target, ok := c.Aliases[name]; ok {
			return target
		}
	}
	return name
}

func defaultModelDir() string {
	if xdg := os.Getenv("XDG_DATA_HOME"); xdg != "" {
		return filepath.Join(xdg, "nollama", "models")
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".local", "share", "nollama", "models")
	}
	return "/var/lib/nollama/models"
}

func cloneFlags(flags map[string]interface{}) map[string]interface{} {
	if len(flags) == 0 {
		return map[string]interface{}{}
	}
	cloned := make(map[string]interface{}, len(flags))
	for k, v := range flags {
		cloned[k] = v
	}
	return cloned
}

// CloneFlags returns a shallow copy of a flag map.
func CloneFlags(flags map[string]interface{}) map[string]interface{} {
	return cloneFlags(flags)
}

// FlagsMapToSlice converts a map of flag names to a deterministic argv slice.
// e.g. {"ctx-size": 131072, "flash-attn": "on"} -> ["--ctx-size", "131072", "--flash-attn", "on"]
func FlagsMapToSlice(flags map[string]interface{}) []string {
	if len(flags) == 0 {
		return nil
	}

	keys := make([]string, 0, len(flags))
	for key := range flags {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	var result []string
	for _, key := range keys {
		flag := "--" + key
		switch val := flags[key].(type) {
		case bool:
			if val {
				result = append(result, flag)
			}
		case string:
			result = append(result, flag, val)
		default:
			result = append(result, flag, fmt.Sprintf("%v", val))
		}
	}
	return result
}
