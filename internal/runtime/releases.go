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
	Name        string `json:"name"`
	DownloadURL string `json:"browser_download_url"`
	Size        int64  `json:"size"`
}

type githubRelease struct {
	TagName string         `json:"tag_name"`
	Assets  []ReleaseAsset `json:"assets"`
}

func fetchGitHubJSON(endpoint string, dst any) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "anvil")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return resp, fmt.Errorf("fetching %s: %s", endpoint, resp.Status)
	}

	if err := json.NewDecoder(resp.Body).Decode(dst); err != nil {
		resp.Body.Close()
		return resp, fmt.Errorf("decoding %s: %w", endpoint, err)
	}

	return resp, nil
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
	var payload githubRelease
	resp, err := fetchGitHubJSON(endpoint, &payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	release := &Release{
		TagName: payload.TagName,
		Assets:  make([]ReleaseAsset, 0, len(payload.Assets)),
	}
	for _, asset := range payload.Assets {
		release.Assets = append(release.Assets, asset)
	}
	return release, nil
}
