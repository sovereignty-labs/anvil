package runtime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInstallPrebuiltRuntimeSelectsAssetByBackend(t *testing.T) {
	releaseTag := "llama-server-b9375"
	tmpDir := t.TempDir()

	type assetFixture struct {
		name string
		data []byte
		size int64
	}

	files := make(map[string][]byte)
	fixtures := make([]assetFixture, 0, 6)
	for _, backend := range []BuildBackend{BuildBackendCUDA, BuildBackendROCm, BuildBackendCPU} {
		archiveName := fmt.Sprintf("%s-linux-amd64-%s.tar.gz", releaseTag, backend)
		archivePath := filepath.Join(tmpDir, archiveName)
		if err := writePrebuiltArchive(t, archivePath, releaseTag, string(backend), string(backend)); err != nil {
			t.Fatal(err)
		}
		archiveData, err := os.ReadFile(archivePath)
		if err != nil {
			t.Fatal(err)
		}
		shaData, err := os.ReadFile(archivePath + ".sha256")
		if err != nil {
			t.Fatal(err)
		}
		files["/files/"+archiveName] = archiveData
		files["/files/"+archiveName+".sha256"] = shaData

		info, err := os.Stat(archivePath)
		if err != nil {
			t.Fatal(err)
		}
		fixtures = append(fixtures, assetFixture{name: archiveName, data: archiveData, size: info.Size()})
		info, err = os.Stat(archivePath + ".sha256")
		if err != nil {
			t.Fatal(err)
		}
		fixtures = append(fixtures, assetFixture{name: archiveName + ".sha256", data: shaData, size: info.Size()})
	}

	var releaseJSON string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/releases":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(releaseJSON))
		default:
			data, ok := files[r.URL.Path]
			if !ok {
				http.NotFound(w, r)
				return
			}
			_, _ = w.Write(data)
		}
	}))
	t.Cleanup(server.Close)

	assets := make([]ReleaseAsset, 0, len(fixtures))
	for _, fx := range fixtures {
		assets = append(assets, ReleaseAsset{
			Name:        fx.name,
			DownloadURL: server.URL + "/files/" + fx.name,
			Size:        fx.size,
		})
	}
	releases := []githubRelease{{TagName: releaseTag, Assets: assets}}
	payload, err := json.Marshal(releases)
	if err != nil {
		t.Fatal(err)
	}
	releaseJSON = string(payload)

	oldURL := sovereigntyLabsReleasesURL
	sovereigntyLabsReleasesURL = server.URL + "/releases"
	t.Cleanup(func() {
		sovereigntyLabsReleasesURL = oldURL
	})

	cases := []struct {
		name         string
		platform     Platform
		wantBackend  BuildBackend
		wantNotFound bool
	}{
		{
			name:        "cuda",
			platform:    Platform{OS: "linux", Arch: "amd64", CUDA: "available"},
			wantBackend: BuildBackendCUDA,
		},
		{
			name:        "rocm",
			platform:    Platform{OS: "linux", Arch: "amd64", ROCm: "available"},
			wantBackend: BuildBackendROCm,
		},
		{
			name:        "cpu",
			platform:    Platform{OS: "linux", Arch: "amd64"},
			wantBackend: BuildBackendCPU,
		},
		{
			name:         "unmatched-platform",
			platform:     Platform{OS: "linux", Arch: "arm64"},
			wantNotFound: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mgr := &Manager{runtimesDir: t.TempDir()}
			info, err := mgr.installPrebuiltRuntime("", tc.platform)
			if tc.wantNotFound {
				if !errors.Is(err, errPrebuiltRuntimeNotFound) {
					t.Fatalf("installPrebuiltRuntime() error = %v, want not-found", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("installPrebuiltRuntime() error = %v", err)
			}
			if info.Name != "llama-b9375" {
				t.Fatalf("installPrebuiltRuntime() name = %q, want llama-b9375", info.Name)
			}
			if info.Source != "prebuilt" {
				t.Fatalf("installPrebuiltRuntime() source = %q, want prebuilt", info.Source)
			}
			if got := mgr.RuntimeBackend(info.Name); got != tc.wantBackend {
				t.Fatalf("RuntimeBackend() = %q, want %q", got, tc.wantBackend)
			}
		})
	}
}

func TestSelectPrebuiltReleasePrefersNewestTag(t *testing.T) {
	releases := []githubRelease{
		{TagName: "llama-server-b9374"},
		{TagName: "llama-server-b9376"},
		{TagName: "llama-server-b9375"},
	}

	got, err := selectPrebuiltRelease(releases, "")
	if err != nil {
		t.Fatalf("selectPrebuiltRelease() error = %v", err)
	}
	if got.TagName != "llama-server-b9376" {
		t.Fatalf("selectPrebuiltRelease() = %q, want llama-server-b9376", got.TagName)
	}
}

func TestPrebuiltArchiveExtractionAndValidation(t *testing.T) {
	tmpDir := t.TempDir()
	releaseTag := "llama-server-b9375"
	archivePath := filepath.Join(tmpDir, fmt.Sprintf("%s-linux-amd64-cuda.tar.gz", releaseTag))
	if err := writePrebuiltArchive(t, archivePath, releaseTag, "cuda", "cuda"); err != nil {
		t.Fatal(err)
	}

	outDir := filepath.Join(tmpDir, "out")
	if err := extractRuntime(archivePath, outDir); err != nil {
		t.Fatalf("extractRuntime() error = %v", err)
	}

	if backend, err := readBackendMarker(outDir); err != nil || backend != "cuda" {
		t.Fatalf("readBackendMarker() = %q, %v; want cuda, nil", backend, err)
	}

	manifest, err := readRuntimeManifest(filepath.Join(outDir, "nollama-runtime.json"))
	if err != nil {
		t.Fatalf("readRuntimeManifest() error = %v", err)
	}
	if manifest.LlamaCPPVersion != "b9375" || manifest.Backend != "cuda" || manifest.OS != "linux" || manifest.Arch != "amd64" {
		t.Fatalf("manifest = %+v, want b9375/cuda/linux/amd64", manifest)
	}

	if err := validatePrebuiltRuntimeBundle(outDir, releaseTag, Platform{OS: "linux", Arch: "amd64"}, BuildBackendCUDA); err != nil {
		t.Fatalf("validatePrebuiltRuntimeBundle() error = %v", err)
	}
}

func TestValidatePrebuiltRuntimeBundleRejectsBackendMismatch(t *testing.T) {
	tmpDir := t.TempDir()
	releaseTag := "llama-server-b9375"
	archivePath := filepath.Join(tmpDir, fmt.Sprintf("%s-linux-amd64-cuda.tar.gz", releaseTag))
	if err := writePrebuiltArchive(t, archivePath, releaseTag, "cuda", "cpu"); err != nil {
		t.Fatal(err)
	}

	outDir := filepath.Join(tmpDir, "out")
	if err := extractRuntime(archivePath, outDir); err != nil {
		t.Fatalf("extractRuntime() error = %v", err)
	}

	err := validatePrebuiltRuntimeBundle(outDir, releaseTag, Platform{OS: "linux", Arch: "amd64"}, BuildBackendCUDA)
	if err == nil {
		t.Fatal("validatePrebuiltRuntimeBundle() expected backend mismatch error")
	}
	if !strings.Contains(err.Error(), "backend") {
		t.Fatalf("validatePrebuiltRuntimeBundle() error = %v, want backend mismatch", err)
	}
}

func writePrebuiltArchive(t *testing.T, archivePath, releaseTag, assetBackend, packagedBackend string) error {
	t.Helper()

	files := map[string][]byte{
		"llama-server":    []byte("binary-" + packagedBackend),
		"libggml-base.so": []byte("ggml-base-" + packagedBackend),
		"libllama.so":     []byte("llama-" + packagedBackend),
		"libmtmd.so.0":    []byte("mtmd-" + packagedBackend),
		"backend":         []byte(packagedBackend + "\n"),
		"nollama-runtime.json": []byte(fmt.Sprintf(
			`{"llamacpp_version":"%s","backend":"%s","os":"linux","arch":"amd64"}`,
			normalizedPrebuiltVersion(releaseTag),
			packagedBackend,
		)),
	}
	writeTarGz(t, archivePath, files)

	sum, err := prebuiltArchiveSHA256(archivePath)
	if err != nil {
		return err
	}
	shaLine := fmt.Sprintf("%s  %s\n", sum, filepath.Base(archivePath))
	return os.WriteFile(archivePath+".sha256", []byte(shaLine), 0o644)
}

func prebuiltArchiveSHA256(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}
