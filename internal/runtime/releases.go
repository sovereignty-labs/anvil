package runtime

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
)

// Release represents a llama.cpp release.
type Release struct {
	TagName string
	Assets  []ReleaseAsset
}

// ReleaseAsset is a downloadable asset in a llama.cpp release.
type ReleaseAsset struct {
	Name        string
	DownloadURL string
	Size        int64
}

type githubRelease struct {
	TagName string `json:"tag_name"`
	Assets  []struct {
		Name        string `json:"name"`
		DownloadURL string `json:"browser_download_url"`
		Size        int64  `json:"size"`
	} `json:"assets"`
}

// FetchLatestRelease fetches the latest llama.cpp release from GitHub.
func FetchLatestRelease() (*Release, error) {
	return fetchRelease("latest", "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
}

// FetchRelease fetches a specific llama.cpp release tag from GitHub.
func FetchRelease(tag string) (*Release, error) {
	return fetchRelease(tag, "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/"+url.PathEscape(tag))
}

func fetchRelease(label, endpoint string) (*Release, error) {
	req, err := http.NewRequest(http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "nollama")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetching GitHub release %s: %s", label, resp.Status)
	}

	var payload githubRelease
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("decoding GitHub release %s: %w", label, err)
	}

	release := &Release{
		TagName: payload.TagName,
		Assets:  make([]ReleaseAsset, 0, len(payload.Assets)),
	}
	for _, asset := range payload.Assets {
		release.Assets = append(release.Assets, ReleaseAsset{
			Name:        asset.Name,
			DownloadURL: asset.DownloadURL,
			Size:        asset.Size,
		})
	}
	return release, nil
}
