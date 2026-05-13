package model

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
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

// ScanDir scans a directory for .gguf files and returns their metadata.
// Does not recurse into subdirectories.
func ScanDir(dir string) ([]ModelInfo, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("scanning model dir %s: %w", dir, err)
	}

	var models []ModelInfo
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasSuffix(strings.ToLower(entry.Name()), ".gguf") {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		info, err := entry.Info()
		if err != nil {
			continue // skip files we can't stat
		}

		mi := ModelInfo{
			Filename:  entry.Name(),
			Path:      path,
			SizeBytes: info.Size(),
		}

		// Try to parse GGUF metadata. Non-fatal if it fails.
		if meta, err := ParseGGUF(path); err == nil {
			mi.Meta = meta
		}

		models = append(models, mi)
	}

	// Sort by filename
	sort.Slice(models, func(i, j int) bool {
		return models[i].Filename < models[j].Filename
	})

	return models, nil
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
