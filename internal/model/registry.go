package model

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// ModelInfo holds metadata about a GGUF file on disk.
type ModelInfo struct {
	// Filename is the base name of the GGUF file.
	Filename string

	// Path is the absolute path to the file.
	Path string

	// SizeBytes is the file size in bytes.
	SizeBytes int64

	// Metadata from the GGUF header (nil if parsing failed).
	Meta *GGUFMetadata
}

// SizeHuman returns a human-readable file size.
func (m *ModelInfo) SizeHuman() string {
	const (
		gb = 1024 * 1024 * 1024
		mb = 1024 * 1024
	)
	if m.SizeBytes >= gb {
		return fmt.Sprintf("%.1f GB", float64(m.SizeBytes)/float64(gb))
	}
	return fmt.Sprintf("%.0f MB", float64(m.SizeBytes)/float64(mb))
}

// cacheEntry stores parsed GGUF metadata with file identity for invalidation.
type cacheEntry struct {
	meta    *GGUFMetadata
	modTime time.Time
	size    int64
}

var (
	metaCache   = make(map[string]cacheEntry)
	metaCacheMu sync.RWMutex
)

// ClearMetadataCache clears the in-memory GGUF metadata cache.
// Exposed for testing.
func ClearMetadataCache() {
	metaCacheMu.Lock()
	metaCache = make(map[string]cacheEntry)
	metaCacheMu.Unlock()
}

// cachedParseGGUF returns cached metadata if the file hasn't changed,
// otherwise parses fresh and updates the cache.
func cachedParseGGUF(path string, modTime time.Time, size int64) (*GGUFMetadata, error) {
	metaCacheMu.RLock()
	if entry, ok := metaCache[path]; ok {
		if entry.modTime.Equal(modTime) && entry.size == size {
			metaCacheMu.RUnlock()
			return entry.meta, nil
		}
	}
	metaCacheMu.RUnlock()

	meta, err := ParseGGUF(path)
	if err != nil {
		return nil, err
	}

	metaCacheMu.Lock()
	metaCache[path] = cacheEntry{
		meta:    meta,
		modTime: modTime,
		size:    size,
	}
	metaCacheMu.Unlock()

	return meta, nil
}

// ScanDir scans a directory for .gguf files and returns their metadata.
// Does not recurse into subdirectories.
func ScanDir(dir string) ([]ModelInfo, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("scanning model dir %s: %w", dir, err)
	}

	type candidate struct {
		name    string
		path    string
		size    int64
		modTime time.Time
	}
	var candidates []candidate
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasSuffix(strings.ToLower(entry.Name()), ".gguf") {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		candidates = append(candidates, candidate{
			name:    entry.Name(),
			path:    filepath.Join(dir, entry.Name()),
			size:    info.Size(),
			modTime: info.ModTime(),
		})
	}

	const maxParallel = 4
	sem := make(chan struct{}, maxParallel)
	results := make([]ModelInfo, len(candidates))
	var wg sync.WaitGroup

	for i, c := range candidates {
		wg.Add(1)
		go func(idx int, c candidate) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			mi := ModelInfo{
				Filename:  c.name,
				Path:      c.path,
				SizeBytes: c.size,
			}
			if meta, err := cachedParseGGUF(c.path, c.modTime, c.size); err == nil {
				meta.QuantName = meta.QuantDisplayName(c.name)
				mi.Meta = meta
			}
			results[idx] = mi
		}(i, c)
	}
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		return results[i].Filename < results[j].Filename
	})

	return results, nil
}

// FuzzyMatchModel finds a loaded model by fuzzy matching against filename stems.
// Strips .gguf extension, case-insensitive comparison.
// Returns the full filename if matched, empty string if not.
func FuzzyMatchModel(name string, available []string) string {
	name = strings.ToLower(strings.TrimSuffix(name, ".gguf"))

	// Exact stem match first
	for _, a := range available {
		stem := strings.ToLower(strings.TrimSuffix(a, ".gguf"))
		if stem == name {
			return a
		}
	}

	// Prefix match
	for _, a := range available {
		stem := strings.ToLower(strings.TrimSuffix(a, ".gguf"))
		if strings.HasPrefix(stem, name) {
			return a
		}
	}

	// Contains match
	for _, a := range available {
		stem := strings.ToLower(strings.TrimSuffix(a, ".gguf"))
		if strings.Contains(stem, name) {
			return a
		}
	}

	return ""
}
