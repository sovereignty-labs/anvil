package pull

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strings"
)

// GGUFFile represents a GGUF file in a HuggingFace repo.
type GGUFFile struct {
	Name        string // e.g. "Qwen3.6-35B-A3B-Q4_K_S.gguf"
	Size        int64  // bytes
	SHA256      string // from LFS metadata
	DownloadURL string // full URL for download
}

type hfTreeEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
	LFS  *struct {
		OID  string `json:"oid"`
		Size int64  `json:"size"`
	} `json:"lfs"`
}

// ListGGUFs fetches the file tree for a HuggingFace repo and returns only .gguf files.
// Uses: GET https://huggingface.co/api/models/{org}/{repo}/tree/main
// If hfToken is non-empty, sends Authorization: Bearer header (for gated models).
func ListGGUFs(org, repo, hfToken string) ([]GGUFFile, error) {
	apiURL := fmt.Sprintf(
		"https://huggingface.co/api/models/%s/%s/tree/main",
		url.PathEscape(org),
		url.PathEscape(repo),
	)

	req, err := http.NewRequest(http.MethodGet, apiURL, nil)
	if err != nil {
		return nil, err
	}
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, httpStatusError(resp.StatusCode, org, repo)
	}

	var entries []hfTreeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, fmt.Errorf("decoding HuggingFace tree for %s/%s: %w", org, repo, err)
	}

	files := make([]GGUFFile, 0, len(entries))
	for _, entry := range entries {
		if entry.Type != "file" {
			continue
		}
		if !strings.HasSuffix(strings.ToLower(entry.Path), ".gguf") {
			continue
		}

		file := GGUFFile{
			Name:        path.Base(entry.Path),
			Size:        entry.Size,
			DownloadURL: downloadURL(org, repo, entry.Path),
		}
		if entry.LFS != nil {
			file.SHA256 = strings.TrimPrefix(entry.LFS.OID, "sha256:")
			if entry.LFS.Size > 0 {
				file.Size = entry.LFS.Size
			}
		}
		files = append(files, file)
	}

	sort.Slice(files, func(i, j int) bool {
		return strings.ToLower(files[i].Name) < strings.ToLower(files[j].Name)
	})

	return files, nil
}

func downloadURL(org, repo, filename string) string {
	return fmt.Sprintf(
		"https://huggingface.co/%s/%s/resolve/main/%s",
		url.PathEscape(org),
		url.PathEscape(repo),
		escapePath(filename),
	)
}

func escapePath(p string) string {
	parts := strings.Split(p, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

func httpStatusError(status int, org, repo string) error {
	switch status {
	case http.StatusNotFound:
		return fmt.Errorf("HuggingFace repo %s/%s not found or private (set HF_TOKEN if this is gated)", org, repo)
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("authentication required for HuggingFace repo %s/%s (set HF_TOKEN)", org, repo)
	default:
		return fmt.Errorf("HuggingFace API returned %d for %s/%s", status, org, repo)
	}
}
