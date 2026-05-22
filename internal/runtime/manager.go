package runtime

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	stdruntime "runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sovereignty-labs/nollama/internal/config"
)

const (
	// DefaultRuntimesDir is the directory name used under the XDG data dir.
	DefaultRuntimesDir = "runtimes"

	activeFileName = "active"
	sourceFileName = ".source"
)

// RuntimeInfo represents an installed runtime.
type RuntimeInfo struct {
	Name    string // e.g. "llama-b9174"
	Path    string // full path to llama-server binary
	Version string // e.g. "b9174" (extracted from name or empty for custom)
	Source  string // "release", "custom"
	Active  bool   // is this the active runtime?
}

// Manager handles runtime lifecycle.
type Manager struct {
	runtimesDir string
}

// NewManager returns a manager rooted at ~/.local/share/nollama/runtimes/.
func NewManager() *Manager {
	dir := defaultRuntimesDir()
	_ = os.MkdirAll(dir, 0o755)
	return &Manager{runtimesDir: dir}
}

func defaultRuntimesDir() string {
	if dir := configuredRuntimesDir(); dir != "" {
		return dir
	}

	varLib := filepath.Join("/var/lib", "nollama", DefaultRuntimesDir)
	home := homeRuntimesDir()

	if runtimeDirHasEntries(varLib) {
		return varLib
	}
	if runtimeDirHasEntries(home) {
		return home
	}
	if _, err := os.Stat(varLib); err == nil {
		return varLib
	}
	if home != "" {
		return home
	}
	return filepath.Join(os.TempDir(), "nollama", DefaultRuntimesDir)
}

func configuredRuntimesDir() string {
	cfgPath := config.FindConfig()
	if cfgPath == "" {
		return ""
	}
	cfg, err := config.Load(cfgPath)
	if err != nil || cfg == nil || strings.TrimSpace(cfg.RuntimesDir) == "" {
		return ""
	}
	dir := strings.TrimSpace(cfg.RuntimesDir)
	if !filepath.IsAbs(dir) {
		abs, err := filepath.Abs(dir)
		if err == nil {
			dir = abs
		}
	}
	return dir
}

func homeRuntimesDir() string {
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".local", "share", "nollama", DefaultRuntimesDir)
	}
	warnHomeUnsetOnce()
	return ""
}

var (
	homeUnsetWarnOnce sync.Once
	stderrWriter      io.Writer = os.Stderr // overridable in tests
)

// warnHomeUnsetOnce emits a single stderr line when HOME is unset, so ops
// running nollama under systemd see why the user-local runtimes path was
// skipped. The resolution chain still falls through to /var/lib and /tmp.
func warnHomeUnsetOnce() {
	homeUnsetWarnOnce.Do(func() {
		fmt.Fprintln(stderrWriter, "warning: HOME not set; skipping user-local runtimes dir. Set runtimes_dir in config or install runtimes to /var/lib/nollama/runtimes.")
	})
}

func runtimeDirHasEntries(dir string) bool {
	if dir == "" {
		return false
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	return len(entries) > 0
}

func (m *Manager) ensureDir() error {
	if m.runtimesDir == "" {
		m.runtimesDir = defaultRuntimesDir()
	}
	return os.MkdirAll(m.runtimesDir, 0o755)
}

func (m *Manager) activeFilePath() string {
	return filepath.Join(m.runtimesDir, activeFileName)
}

func (m *Manager) runtimeDir(name string) (string, error) {
	if err := validateRuntimeName(name); err != nil {
		return "", err
	}
	return filepath.Join(m.runtimesDir, name), nil
}

func (m *Manager) runtimeBinaryPath(name string) (string, error) {
	dir, err := m.runtimeDir(name)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, runtimeBinaryName()), nil
}

// Install downloads and installs a pre-built llama-server binary.
func (m *Manager) Install(version string) (*RuntimeInfo, error) {
	if err := m.ensureDir(); err != nil {
		return nil, fmt.Errorf("prepare runtimes dir: %w", err)
	}

	platform := DetectPlatform()
	fmt.Fprintf(os.Stderr, "Detecting platform... %s %s", prettyOS(platform.OS), prettyArch(platform.Arch))
	if platform.CUDA != "" {
		fmt.Fprint(os.Stderr, ", NVIDIA GPU (CUDA available)\n")
	} else {
		fmt.Fprint(os.Stderr, ", CPU-only\n")
	}

	var release *Release
	var err error
	if version == "" {
		fmt.Fprintln(os.Stderr, "Fetching latest llama.cpp release...")
		release, err = FetchLatestRelease()
	} else {
		fmt.Fprintf(os.Stderr, "Fetching llama.cpp release %s...\n", version)
		release, err = FetchRelease(version)
	}
	if err != nil {
		return nil, err
	}

	fmt.Fprintf(os.Stderr, "Release: %s\n", release.TagName)

	asset, err := SelectAsset(release.Assets, platform)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(os.Stderr, "Selected: %s (%s)\n", asset.Name, humanBytes(asset.Size))

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

	archivePath := filepath.Join(workDir, asset.Name)
	fmt.Fprintln(os.Stderr, "\nDownloading...")
	if err := downloadWithProgress(asset.DownloadURL, archivePath, asset.Size); err != nil {
		return nil, err
	}

	fmt.Fprintln(os.Stderr, "\nExtracting runtime...")
	if err := extractRuntime(archivePath, workDir); err != nil {
		return nil, err
	}
	binaryPath := filepath.Join(workDir, runtimeBinaryName())
	if _, err := os.Stat(binaryPath); err != nil {
		return nil, fmt.Errorf("llama-server missing from extracted archive %s: %w", asset.Name, err)
	}
	if err := chmodRuntimeArtifacts(workDir); err != nil {
		return nil, err
	}
	// Archive is no longer needed; reclaim the disk.
	_ = os.Remove(archivePath)
	if err := writeSourceMarker(workDir, "release"); err != nil {
		return nil, err
	}

	if err := os.RemoveAll(finalDir); err != nil {
		return nil, fmt.Errorf("remove existing runtime %s: %w", finalDir, err)
	}
	if err := os.Rename(workDir, finalDir); err != nil {
		return nil, fmt.Errorf("finalize runtime install: %w", err)
	}
	cleanup = false

	if err := m.Use(runtimeName); err != nil {
		return nil, err
	}

	info := RuntimeInfo{
		Name:    runtimeName,
		Path:    filepath.Join(finalDir, runtimeBinaryName()),
		Version: versionFromRuntimeName(runtimeName),
		Source:  "release",
		Active:  true,
	}
	fmt.Fprintf(os.Stderr, "Installed: %s\n", info.Path)
	fmt.Fprintf(os.Stderr, "Active runtime: %s ✓\n", info.Name)
	return &info, nil
}

// BuildOpts configures a from-source runtime build.
type BuildOpts struct {
	Repo   string // git URL (default: github.com/ggml-org/llama.cpp)
	Branch string // optional branch / tag to checkout
	Name   string // runtime name (default: derived from repo)
}

// Build clones llama.cpp (or a fork), runs cmake + cmake --build, then copies
// llama-server and its shared libraries into a new runtime directory and
// marks the runtime active.
func (m *Manager) Build(opts BuildOpts) (*RuntimeInfo, error) {
	if err := m.ensureDir(); err != nil {
		return nil, fmt.Errorf("prepare runtimes dir: %w", err)
	}

	repo := strings.TrimSpace(opts.Repo)
	if repo == "" {
		repo = defaultBuildRepo
	}
	name := strings.TrimSpace(opts.Name)
	if name == "" {
		name = deriveBuildRuntimeName(repo, opts.Branch)
	}
	if err := validateRuntimeName(name); err != nil {
		return nil, err
	}

	platform := DetectPlatform()
	tools, err := checkBuildTools(platform)
	if err != nil {
		return nil, err
	}
	printBuildTools(tools)

	srcDir, err := os.MkdirTemp("", "nollama-build-")
	if err != nil {
		return nil, fmt.Errorf("create build temp dir: %w", err)
	}
	defer os.RemoveAll(srcDir)

	cloneArgs := []string{"clone", "--depth", "1"}
	if b := strings.TrimSpace(opts.Branch); b != "" {
		cloneArgs = append(cloneArgs, "--branch", b)
	}
	cloneArgs = append(cloneArgs, repo, srcDir)
	fmt.Fprintf(os.Stderr, "\nCloning %s...\n", repo)
	if err := runBuildCmd(srcDir, "git", cloneArgs...); err != nil {
		return nil, fmt.Errorf("git clone: %w", err)
	}

	cmakeArgs := []string{
		"-B", "build",
		"-DBUILD_SHARED_LIBS=ON",
		"-DLLAMA_BUILD_TESTS=OFF",
		"-DLLAMA_BUILD_EXAMPLES=OFF",
		"-DLLAMA_BUILD_SERVER=ON",
	}
	if platform.CUDA != "" && tools.nvcc != "" {
		cmakeArgs = append(cmakeArgs, "-DGGML_CUDA=ON")
	}
	fmt.Fprintf(os.Stderr, "\nConfiguring: cmake %s\n", strings.Join(cmakeArgs, " "))
	if err := runBuildCmd(srcDir, "cmake", cmakeArgs...); err != nil {
		return nil, fmt.Errorf("cmake configure: %w", err)
	}

	jobs := fmt.Sprintf("%d", stdruntime.NumCPU())
	fmt.Fprintf(os.Stderr, "\nBuilding (-j%s)...\n", jobs)
	if err := runBuildCmd(srcDir, "cmake", "--build", "build", "--config", "Release", "-j", jobs); err != nil {
		return nil, fmt.Errorf("cmake build: %w", err)
	}

	binarySrc, libs, err := findBuildArtifacts(filepath.Join(srcDir, "build"))
	if err != nil {
		return nil, err
	}

	finalDir, err := m.runtimeDir(name)
	if err != nil {
		return nil, err
	}
	workDir, err := os.MkdirTemp(m.runtimesDir, "."+name+".")
	if err != nil {
		return nil, fmt.Errorf("create temp runtime dir: %w", err)
	}
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.RemoveAll(workDir)
		}
	}()

	if err := copyFile(binarySrc, filepath.Join(workDir, runtimeBinaryName())); err != nil {
		return nil, err
	}
	for _, lib := range libs {
		if err := copyFile(lib, filepath.Join(workDir, filepath.Base(lib))); err != nil {
			return nil, err
		}
	}
	if err := chmodRuntimeArtifacts(workDir); err != nil {
		return nil, err
	}
	if err := writeSourceMarker(workDir, "build"); err != nil {
		return nil, err
	}

	if err := os.RemoveAll(finalDir); err != nil {
		return nil, fmt.Errorf("remove existing runtime %s: %w", finalDir, err)
	}
	if err := os.Rename(workDir, finalDir); err != nil {
		return nil, fmt.Errorf("finalize build runtime: %w", err)
	}
	cleanup = false

	if err := m.Use(name); err != nil {
		return nil, err
	}

	info := RuntimeInfo{
		Name:    name,
		Path:    filepath.Join(finalDir, runtimeBinaryName()),
		Version: "",
		Source:  "build",
		Active:  true,
	}
	fmt.Fprintf(os.Stderr, "Installed: %s\n", info.Path)
	fmt.Fprintf(os.Stderr, "Active runtime: %s ✓\n", info.Name)
	return &info, nil
}

// List scans installed runtimes.
func (m *Manager) List() ([]RuntimeInfo, error) {
	if err := m.ensureDir(); err != nil {
		return nil, fmt.Errorf("prepare runtimes dir: %w", err)
	}

	entries, err := os.ReadDir(m.runtimesDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("read runtimes dir: %w", err)
	}

	activeName, _ := m.readActiveName()

	infos := make([]RuntimeInfo, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}

		dir := filepath.Join(m.runtimesDir, name)
		bin, ok, err := m.findBinary(dir)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}

		info := RuntimeInfo{
			Name:    name,
			Path:    bin,
			Version: versionFromRuntimeName(name),
			Source:  readSourceMarker(dir),
			Active:  name == activeName,
		}
		if info.Source == "" {
			info.Source = sourceFromRuntimeName(name)
		}
		infos = append(infos, info)
	}

	sort.Slice(infos, func(i, j int) bool {
		return infos[i].Name < infos[j].Name
	})
	return infos, nil
}

// Use sets the active runtime.
func (m *Manager) Use(name string) error {
	if err := m.ensureDir(); err != nil {
		return fmt.Errorf("prepare runtimes dir: %w", err)
	}
	if err := validateRuntimeName(name); err != nil {
		return err
	}

	if _, ok, err := m.findBinary(filepath.Join(m.runtimesDir, name)); err != nil {
		return err
	} else if !ok {
		return fmt.Errorf("runtime %q does not exist or is missing llama-server", name)
	}

	return os.WriteFile(m.activeFilePath(), []byte(name+"\n"), 0o644)
}

// Add registers an external llama-server binary.
func (m *Manager) Add(name, binaryPath string) error {
	if err := m.ensureDir(); err != nil {
		return fmt.Errorf("prepare runtimes dir: %w", err)
	}
	if err := validateRuntimeName(name); err != nil {
		return err
	}

	srcInfo, err := os.Stat(binaryPath)
	if err != nil {
		return fmt.Errorf("stat llama-server binary %s: %w", binaryPath, err)
	}
	if srcInfo.IsDir() {
		return fmt.Errorf("%s is a directory, expected a llama-server binary", binaryPath)
	}

	destDir, err := m.runtimeDir(name)
	if err != nil {
		return err
	}
	workDir, err := os.MkdirTemp(m.runtimesDir, "."+name+".")
	if err != nil {
		return fmt.Errorf("create temp runtime dir: %w", err)
	}
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.RemoveAll(workDir)
		}
	}()

	destBinary := filepath.Join(workDir, runtimeBinaryName())
	if err := copyFile(binaryPath, destBinary); err != nil {
		return err
	}
	if err := os.Chmod(destBinary, 0o755); err != nil {
		return fmt.Errorf("chmod copied llama-server: %w", err)
	}
	if err := writeSourceMarker(workDir, "custom"); err != nil {
		return err
	}

	if err := os.RemoveAll(destDir); err != nil {
		return fmt.Errorf("remove existing runtime %s: %w", destDir, err)
	}
	if err := os.Rename(workDir, destDir); err != nil {
		return fmt.Errorf("install custom runtime: %w", err)
	}
	cleanup = false
	return nil
}

// Resolve returns the active llama-server binary path.
func (m *Manager) Resolve() (string, error) {
	if err := m.ensureDir(); err != nil {
		return "", fmt.Errorf("prepare runtimes dir: %w", err)
	}

	activeName, err := m.readActiveName()
	if err != nil {
		return "", err
	}
	if activeName == "" {
		return "", fmt.Errorf("no active runtime found. Run `nollama runtime install` or `nollama runtime use <name>`")
	}

	bin, ok, err := m.findBinary(filepath.Join(m.runtimesDir, activeName))
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("active runtime %q has no llama-server binary", activeName)
	}
	return bin, nil
}

// ActiveName returns the configured active runtime name, if any.
func (m *Manager) ActiveName() (string, error) {
	if err := m.ensureDir(); err != nil {
		return "", fmt.Errorf("prepare runtimes dir: %w", err)
	}
	return m.readActiveName()
}

func (m *Manager) readActiveName() (string, error) {
	data, err := os.ReadFile(m.activeFilePath())
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", fmt.Errorf("read active runtime: %w", err)
	}
	return strings.TrimSpace(string(data)), nil
}

func (m *Manager) findBinary(dir string) (string, bool, error) {
	for _, candidate := range binaryCandidates() {
		path := filepath.Join(dir, candidate)
		info, err := os.Stat(path)
		if err == nil && !info.IsDir() {
			return path, true, nil
		}
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return "", false, fmt.Errorf("stat runtime binary %s: %w", path, err)
		}
	}
	return "", false, nil
}

func binaryCandidates() []string {
	if stdruntime.GOOS == "windows" {
		return []string{"llama-server.exe", "llama-server"}
	}
	return []string{"llama-server", "llama-server.exe"}
}

func runtimeBinaryName() string {
	if stdruntime.GOOS == "windows" {
		return "llama-server.exe"
	}
	return "llama-server"
}

func validateRuntimeName(name string) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("runtime name cannot be empty")
	}
	if strings.ContainsAny(name, `/\`) {
		return fmt.Errorf("invalid runtime name %q", name)
	}
	return nil
}

func runtimeNameForTag(tag string) string {
	tag = strings.TrimSpace(tag)
	tag = strings.TrimPrefix(tag, "v")
	if tag == "" {
		tag = "latest"
	}
	return "llama-" + tag
}

func versionFromRuntimeName(name string) string {
	if strings.HasPrefix(name, "llama-") {
		return strings.TrimPrefix(name, "llama-")
	}
	return ""
}

func sourceFromRuntimeName(name string) string {
	if strings.HasPrefix(name, "llama-") {
		return "release"
	}
	return "custom"
}

func readSourceMarker(dir string) string {
	data, err := os.ReadFile(filepath.Join(dir, sourceFileName))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func writeSourceMarker(dir, source string) error {
	return os.WriteFile(filepath.Join(dir, sourceFileName), []byte(source+"\n"), 0o644)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("open source binary %s: %w", src, err)
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("create destination binary %s: %w", dst, err)
	}
	defer func() {
		_ = out.Close()
	}()

	if _, err := io.Copy(out, in); err != nil {
		return fmt.Errorf("copy llama-server binary: %w", err)
	}
	return nil
}

func humanBytes(bytes int64) string {
	const (
		mb = 1024 * 1024
		gb = 1024 * mb
	)
	switch {
	case bytes >= gb:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(gb))
	case bytes >= mb:
		return fmt.Sprintf("%.0f MB", float64(bytes)/float64(mb))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

func prettyOS(osName string) string {
	switch osName {
	case "darwin":
		return "macOS"
	case "windows":
		return "Windows"
	default:
		return strings.Title(osName)
	}
}

func prettyArch(arch string) string {
	switch arch {
	case "amd64":
		return "x64"
	case "arm64":
		return "arm64"
	default:
		return arch
	}
}

func downloadWithProgress(url, dest string, total int64) error {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", url, resp.Status)
	}

	if total <= 0 && resp.ContentLength > 0 {
		total = resp.ContentLength
	}

	out, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("create archive %s: %w", dest, err)
	}
	defer func() {
		_ = out.Close()
	}()

	start := time.Now()
	writer := &countingWriter{
		w: out,
		onChange: func(downloaded int64) {
			if total <= 0 {
				fmt.Fprintf(os.Stderr, "\r  %s downloaded", humanBytes(downloaded))
				return
			}
			elapsed := time.Since(start).Seconds()
			speed := float64(0)
			if elapsed > 0 {
				speed = float64(downloaded) / elapsed
			}
			fmt.Fprintf(os.Stderr, "\r  %s / %s  %3d%%  %s/s  ETA %s",
				humanBytes(downloaded),
				humanBytes(total),
				downloadPercent(downloaded, total),
				humanRate(speed),
				downloadETA(downloaded, total, speed),
			)
		},
	}

	if _, err := io.Copy(writer, resp.Body); err != nil {
		return fmt.Errorf("download archive: %w", err)
	}
	fmt.Fprintln(os.Stderr)
	return nil
}

type countingWriter struct {
	w        io.Writer
	count    int64
	onChange func(int64)
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.count += int64(n)
	if cw.onChange != nil {
		cw.onChange(cw.count)
	}
	return n, err
}

func downloadPercent(downloaded, total int64) int {
	if total <= 0 {
		return 0
	}
	if downloaded >= total {
		return 100
	}
	return int((downloaded * 100) / total)
}

func downloadETA(downloaded, total int64, speed float64) string {
	if total <= 0 || speed <= 0 {
		return "--"
	}
	remaining := float64(total-downloaded) / speed
	if remaining < 1 {
		return "<1s"
	}
	return fmt.Sprintf("%.0fs", remaining)
}

func humanRate(rate float64) string {
	const (
		mb = 1024.0 * 1024.0
		gb = 1024.0 * mb
	)
	switch {
	case rate >= gb:
		return fmt.Sprintf("%.0f GB", rate/gb)
	case rate >= mb:
		return fmt.Sprintf("%.0f MB", rate/mb)
	default:
		return fmt.Sprintf("%.0f B", rate)
	}
}

// extractRuntime extracts every regular file from the archive into destDir,
// flattening one level of top-level directory nesting (llama.cpp release
// tarballs wrap everything under llama-b9275/). On success, llama-server and
// every shared library land next to each other so the dynamic linker can
// find them via LD_LIBRARY_PATH=<destDir>.
func extractRuntime(archivePath, destDir string) error {
	lower := strings.ToLower(archivePath)
	switch {
	case strings.HasSuffix(lower, ".zip"):
		return extractRuntimeFromZip(archivePath, destDir)
	case strings.HasSuffix(lower, ".tar.gz"):
		return extractRuntimeFromTarGz(archivePath, destDir)
	default:
		return fmt.Errorf("unsupported archive format: %s", archivePath)
	}
}

// stripTopDir removes the first path component from name. "llama-b9275/foo/bar"
// → "foo/bar"; "foo" → "foo" (no top-level dir to strip).
func stripTopDir(name string) string {
	name = strings.TrimLeft(name, "/")
	idx := strings.Index(name, "/")
	if idx < 0 {
		return name
	}
	return name[idx+1:]
}

func extractRuntimeFromZip(archivePath, destDir string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("open zip archive %s: %w", archivePath, err)
	}
	defer r.Close()

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return fmt.Errorf("create extraction dir: %w", err)
	}

	count := 0
	for _, file := range r.File {
		if file.FileInfo().IsDir() {
			continue
		}
		stripped := stripTopDir(file.Name)
		if stripped == "" || strings.Contains(stripped, "..") {
			continue
		}
		outPath := filepath.Join(destDir, filepath.Base(stripped))
		if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
			return fmt.Errorf("create dir for %s: %w", outPath, err)
		}
		rc, err := file.Open()
		if err != nil {
			return fmt.Errorf("open %s in archive: %w", file.Name, err)
		}
		if err := writeArchiveEntry(outPath, rc); err != nil {
			rc.Close()
			return err
		}
		rc.Close()
		count++
	}
	if count == 0 {
		return fmt.Errorf("no files extracted from %s", archivePath)
	}
	return nil
}

func extractRuntimeFromTarGz(archivePath, destDir string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("open tar.gz archive %s: %w", archivePath, err)
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("open gzip archive %s: %w", archivePath, err)
	}
	defer gzr.Close()

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return fmt.Errorf("create extraction dir: %w", err)
	}

	tr := tar.NewReader(gzr)
	count := 0
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read tar archive %s: %w", archivePath, err)
		}
		if hdr.FileInfo().IsDir() {
			continue
		}
		stripped := stripTopDir(hdr.Name)
		if stripped == "" || strings.Contains(stripped, "..") {
			continue
		}
		outPath := filepath.Join(destDir, filepath.Base(stripped))
		if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
			return fmt.Errorf("create dir for %s: %w", outPath, err)
		}
		if err := writeArchiveEntry(outPath, tr); err != nil {
			return err
		}
		count++
	}
	if count == 0 {
		return fmt.Errorf("no files extracted from %s", archivePath)
	}
	return nil
}

// chmodRuntimeArtifacts makes the llama-server binary and any .so files in dir
// executable / readable (0o755). Other files are left at whatever mode the
// archive stored.
func chmodRuntimeArtifacts(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("read runtime dir: %w", err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		base := strings.ToLower(name)
		if base == "llama-server" || base == "llama-server.exe" || strings.Contains(base, ".so") {
			if err := os.Chmod(filepath.Join(dir, name), 0o755); err != nil {
				return fmt.Errorf("chmod %s: %w", name, err)
			}
		}
	}
	return nil
}

func writeArchiveEntry(destPath string, src io.Reader) error {
	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return fmt.Errorf("create extraction dir: %w", err)
	}

	dst, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("create llama-server binary %s: %w", destPath, err)
	}
	defer func() {
		_ = dst.Close()
	}()

	if _, err := io.Copy(dst, src); err != nil {
		return fmt.Errorf("extract llama-server binary: %w", err)
	}
	return nil
}
