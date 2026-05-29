package runtime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

var sovereigntyLabsReleasesURL = "https://api.github.com/repos/sovereignty-labs/nollama/releases"

var errPrebuiltRuntimeNotFound = errors.New("prebuilt runtime not found")

type prebuiltRuntimeManifest struct {
	LlamaCPPVersion string `json:"llamacpp_version"`
	Backend         string `json:"backend"`
	OS              string `json:"os"`
	Arch            string `json:"arch"`
}

func (m *Manager) installPrebuiltRuntime(version string, platform Platform) (*RuntimeInfo, error) {
	if platform.OS != "linux" {
		return nil, errPrebuiltRuntimeNotFound
	}

	releases, err := fetchSovereigntyLabReleases()
	if err != nil {
		return nil, err
	}

	release, err := selectPrebuiltRelease(releases, version)
	if err != nil {
		return nil, err
	}

	backend := prebuiltBackendForPlatform(platform)
	tarballName := fmt.Sprintf("%s-linux-%s-%s.tar.gz", release.TagName, platform.Arch, backend)
	shaName := tarballName + ".sha256"

	asset, ok := findPrebuiltAsset(release.Assets, tarballName)
	if !ok {
		return nil, errPrebuiltRuntimeNotFound
	}
	shaAsset, ok := findPrebuiltAsset(release.Assets, shaName)
	if !ok {
		return nil, errPrebuiltRuntimeNotFound
	}

	runtimeName := runtimeNameForTag(release.TagName)
	finalDir, err := m.runtimeDir(runtimeName)
	if err != nil {
		return nil, err
	}
	workDir, err := os.MkdirTemp(m.runtimesDir, "."+runtimeName+".")
	if err != nil {
		return nil, fmt.Errorf("create temp runtime dir: %w", err)
	}
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.RemoveAll(workDir)
		}
	}()

	fmt.Fprintf(os.Stderr, "Using pre-built %s runtime from sovereignty-labs release %s\n", backendDisplayName(backend), release.TagName)

	archivePath := filepath.Join(workDir, asset.Name)
	checksumPath := filepath.Join(workDir, shaAsset.Name)

	fmt.Fprintln(os.Stderr, "\nDownloading...")
	if err := downloadWithProgress(asset.DownloadURL, archivePath, asset.Size); err != nil {
		return nil, errPrebuiltRuntimeNotFound
	}
	if err := downloadWithProgress(shaAsset.DownloadURL, checksumPath, shaAsset.Size); err != nil {
		return nil, errPrebuiltRuntimeNotFound
	}

	expectedSHA, err := readSHA256Sidecar(checksumPath)
	if err != nil {
		return nil, err
	}
	if err := verifySHA256File(archivePath, expectedSHA); err != nil {
		return nil, err
	}

	fmt.Fprintln(os.Stderr, "\nExtracting runtime...")
	if err := extractRuntime(archivePath, workDir); err != nil {
		return nil, err
	}

	if err := validatePrebuiltRuntimeBundle(workDir, release.TagName, platform, backend); err != nil {
		return nil, err
	}

	if err := writeSourceMarker(workDir, "prebuilt"); err != nil {
		return nil, err
	}
	if err := chmodRuntimeArtifacts(workDir); err != nil {
		return nil, err
	}

	if err := os.RemoveAll(finalDir); err != nil {
		return nil, fmt.Errorf("remove existing runtime %s: %w", finalDir, err)
	}
	if err := os.Rename(workDir, finalDir); err != nil {
		return nil, fmt.Errorf("finalize prebuilt runtime install: %w", err)
	}
	cleanup = false

	if err := m.Use(runtimeName); err != nil {
		return nil, err
	}

	info := RuntimeInfo{
		Name:    runtimeName,
		Path:    filepath.Join(finalDir, runtimeBinaryName()),
		Version: versionFromRuntimeName(runtimeName),
		Source:  "prebuilt",
		Active:  true,
	}
	fmt.Fprintf(os.Stderr, "Installed: %s\n", info.Path)
	fmt.Fprintf(os.Stderr, "Active runtime: %s ✓\n", info.Name)
	return &info, nil
}

func fetchSovereigntyLabReleases() ([]githubRelease, error) {
	var releases []githubRelease
	resp, err := fetchGitHubJSON(sovereigntyLabsReleasesURL, &releases)
	if resp != nil {
		defer resp.Body.Close()
	}
	if err != nil {
		return nil, errPrebuiltRuntimeNotFound
	}
	return releases, nil
}

func selectPrebuiltRelease(releases []githubRelease, requestedVersion string) (*githubRelease, error) {
	candidates := make([]githubRelease, 0, len(releases))
	for _, release := range releases {
		if strings.HasPrefix(release.TagName, "llama-server-") {
			candidates = append(candidates, release)
		}
	}
	if len(candidates) == 0 {
		return nil, errPrebuiltRuntimeNotFound
	}

	if requestedVersion = normalizedPrebuiltVersion(requestedVersion); requestedVersion != "" {
		wantedTag := "llama-server-" + requestedVersion
		for i := range candidates {
			if candidates[i].TagName == wantedTag {
				return &candidates[i], nil
			}
		}
		return nil, errPrebuiltRuntimeNotFound
	}

	sort.Slice(candidates, func(i, j int) bool {
		return comparePrebuiltTags(candidates[i].TagName, candidates[j].TagName) > 0
	})
	return &candidates[0], nil
}

func findPrebuiltAsset(assets []ReleaseAsset, name string) (ReleaseAsset, bool) {
	for _, asset := range assets {
		if asset.Name == name {
			return asset, true
		}
	}
	return ReleaseAsset{}, false
}

func prebuiltBackendForPlatform(platform Platform) BuildBackend {
	switch {
	case platform.CUDA != "":
		return BuildBackendCUDA
	case platform.ROCm != "":
		return BuildBackendROCm
	default:
		return BuildBackendCPU
	}
}

func normalizedPrebuiltVersion(tag string) string {
	tag = strings.TrimSpace(tag)
	tag = strings.TrimPrefix(tag, "llama-server-")
	tag = strings.TrimPrefix(tag, "v")
	return tag
}

func comparePrebuiltTags(a, b string) int {
	ap := prebuiltTagParts(a)
	bp := prebuiltTagParts(b)
	for i := 0; i < len(ap) && i < len(bp); i++ {
		switch {
		case ap[i] < bp[i]:
			return -1
		case ap[i] > bp[i]:
			return 1
		}
	}
	switch {
	case len(ap) < len(bp):
		return -1
	case len(ap) > len(bp):
		return 1
	default:
		return strings.Compare(a, b)
	}
}

func prebuiltTagParts(tag string) []uint64 {
	tag = normalizedPrebuiltVersion(tag)
	if tag == "" {
		return nil
	}

	parts := make([]uint64, 0, 4)
	token := strings.Builder{}
	flush := func() {
		if token.Len() == 0 {
			return
		}
		n, err := strconv.ParseUint(token.String(), 10, 64)
		if err == nil {
			parts = append(parts, n)
		}
		token.Reset()
	}
	for _, r := range tag {
		if r >= '0' && r <= '9' {
			token.WriteRune(r)
			continue
		}
		flush()
	}
	flush()
	return parts
}

func readRuntimeManifest(path string) (*prebuiltRuntimeManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read prebuilt runtime manifest: %w", err)
	}

	var manifest prebuiltRuntimeManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("parse prebuilt runtime manifest: %w", err)
	}
	return &manifest, nil
}

func validatePrebuiltRuntimeBundle(dir, releaseTag string, platform Platform, backend BuildBackend) error {
	manifest, err := readRuntimeManifest(filepath.Join(dir, "nollama-runtime.json"))
	if err != nil {
		return err
	}
	if manifest.LlamaCPPVersion != "" && manifest.LlamaCPPVersion != normalizedPrebuiltVersion(releaseTag) {
		return fmt.Errorf("prebuilt manifest version %q does not match release %q", manifest.LlamaCPPVersion, releaseTag)
	}
	if manifest.OS != "" && manifest.OS != "linux" {
		return fmt.Errorf("prebuilt manifest OS %q does not match linux", manifest.OS)
	}
	if manifest.Arch != "" && manifest.Arch != platform.Arch {
		return fmt.Errorf("prebuilt manifest arch %q does not match %q", manifest.Arch, platform.Arch)
	}

	backendFile, err := readBackendMarker(dir)
	if err != nil {
		return err
	}
	if backendFile != string(backend) {
		return fmt.Errorf("prebuilt backend %q does not match requested backend %q", backendFile, backend)
	}
	if manifest.Backend != "" && manifest.Backend != backendFile {
		return fmt.Errorf("prebuilt manifest backend %q does not match backend file %q", manifest.Backend, backendFile)
	}
	return nil
}

func readBackendMarker(dir string) (string, error) {
	data, err := os.ReadFile(filepath.Join(dir, "backend"))
	if err != nil {
		return "", fmt.Errorf("read prebuilt backend marker: %w", err)
	}
	return strings.TrimSpace(string(data)), nil
}

func readSHA256Sidecar(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read sha256 sidecar: %w", err)
	}
	fields := strings.Fields(string(data))
	if len(fields) == 0 {
		return "", fmt.Errorf("sha256 sidecar %s is empty", path)
	}
	return fields[0], nil
}

func verifySHA256File(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open archive for checksum: %w", err)
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return fmt.Errorf("hash archive: %w", err)
	}
	got := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(got, want) {
		return fmt.Errorf("sha256 mismatch for %s: got %s, want %s", path, got, want)
	}
	return nil
}
