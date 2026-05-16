package federation

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/hirdforge/nollama/internal/config"
	"gopkg.in/yaml.v3"
)

// RemoteEntry is a CLI-managed remote node entry.
type RemoteEntry struct {
	URL string `yaml:"url"`
}

// RemoteRegistry stores CLI-managed remote nodes.
type RemoteRegistry struct {
	Remotes map[string]RemoteEntry `yaml:"remotes"`
}

// DefaultRegistryPath returns the default CLI registry path.
func DefaultRegistryPath() string {
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		return filepath.Join(xdg, "nollama", "remotes.yaml")
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".config", "nollama", "remotes.yaml")
	}
	return filepath.Join(".config", "nollama", "remotes.yaml")
}

// LoadRegistry loads a registry from disk. Missing files yield an empty registry.
func LoadRegistry(path string) (*RemoteRegistry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &RemoteRegistry{Remotes: map[string]RemoteEntry{}}, nil
		}
		return nil, fmt.Errorf("reading registry %s: %w", path, err)
	}

	registry := &RemoteRegistry{}
	if err := yaml.Unmarshal(data, registry); err != nil {
		return nil, fmt.Errorf("parsing registry %s: %w", path, err)
	}
	if registry.Remotes == nil {
		registry.Remotes = map[string]RemoteEntry{}
	}

	return registry, nil
}

// Save writes the registry to disk.
func (r *RemoteRegistry) Save(path string) error {
	if r == nil {
		r = &RemoteRegistry{}
	}
	if r.Remotes == nil {
		r.Remotes = map[string]RemoteEntry{}
	}

	data, err := yaml.Marshal(r)
	if err != nil {
		return fmt.Errorf("marshaling registry: %w", err)
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("creating registry directory %s: %w", filepath.Dir(path), err)
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("writing registry %s: %w", path, err)
	}

	return nil
}

// Add inserts a new remote.
func (r *RemoteRegistry) Add(name, rawURL string) error {
	if r.Remotes == nil {
		r.Remotes = map[string]RemoteEntry{}
	}
	if _, exists := r.Remotes[name]; exists {
		return fmt.Errorf("remote %q already exists", name)
	}

	normalized, err := normalizeRemoteURL(rawURL)
	if err != nil {
		return err
	}

	r.Remotes[name] = RemoteEntry{URL: normalized}
	return nil
}

// Remove deletes a remote.
func (r *RemoteRegistry) Remove(name string) error {
	if r.Remotes == nil {
		r.Remotes = map[string]RemoteEntry{}
	}
	if _, exists := r.Remotes[name]; !exists {
		return fmt.Errorf("remote %q does not exist", name)
	}

	delete(r.Remotes, name)
	return nil
}

// List returns the registry entries.
func (r *RemoteRegistry) List() map[string]RemoteEntry {
	if r == nil || len(r.Remotes) == 0 {
		return map[string]RemoteEntry{}
	}

	out := make(map[string]RemoteEntry, len(r.Remotes))
	for name, entry := range r.Remotes {
		out[name] = entry
	}
	return out
}

// PingRemote checks the remote health endpoint and returns the request latency.
func PingRemote(rawURL string) (time.Duration, error) {
	normalized, err := normalizeRemoteURL(rawURL)
	if err != nil {
		return 0, err
	}

	u, err := url.Parse(normalized)
	if err != nil {
		return 0, fmt.Errorf("parsing remote URL %q: %w", rawURL, err)
	}

	u.Path = strings.TrimRight(u.Path, "/") + "/health"
	if u.Path == "/health" && strings.HasSuffix(normalized, "/health") {
		u.Path = "/health"
	}

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, u.String(), nil)
	if err != nil {
		return 0, fmt.Errorf("building health request for %q: %w", rawURL, err)
	}

	client := &http.Client{Timeout: 5 * time.Second}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	latency := time.Since(start)

	if resp.StatusCode != http.StatusOK {
		return latency, fmt.Errorf("health check returned %s", resp.Status)
	}

	var payload struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return latency, fmt.Errorf("decoding health response: %w", err)
	}
	if payload.Status != "ok" {
		return latency, fmt.Errorf("health check status %q", payload.Status)
	}

	return latency, nil
}

// MergeRemotes merges CLI registry remotes with config remotes.
func MergeRemotes(registry *RemoteRegistry, cfgRemotes map[string]config.Remote) map[string]string {
	merged := map[string]string{}
	if registry != nil {
		for name, entry := range registry.Remotes {
			merged[name] = entry.URL
		}
	}
	for name, entry := range cfgRemotes {
		merged[name] = entry.URL
	}
	return merged
}

func normalizeRemoteURL(rawURL string) (string, error) {
	trimmed := strings.TrimRight(strings.TrimSpace(rawURL), "/")
	if trimmed == "" {
		return "", fmt.Errorf("remote URL cannot be empty")
	}

	u, err := url.Parse(trimmed)
	if err != nil {
		return "", fmt.Errorf("parsing remote URL %q: %w", rawURL, err)
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("remote URL %q must include scheme and host", rawURL)
	}

	return trimmed, nil
}
