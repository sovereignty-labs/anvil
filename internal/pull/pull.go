package pull

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// PullSpec is the parsed form of "org/repo:quant".
type PullSpec struct {
	Org   string
	Repo  string
	Quant string
}

// ParseSpec parses "org/repo:quant" into a PullSpec.
// Returns error if format is invalid (missing org, repo, or quant).
func ParseSpec(spec string) (PullSpec, error) {
	repoPart, quant, ok := strings.Cut(spec, ":")
	if !ok {
		return PullSpec{}, fmt.Errorf("invalid pull spec %q: expected org/repo:quant", spec)
	}
	if quant == "" {
		return PullSpec{}, fmt.Errorf("invalid pull spec %q: quant filter is required after ':'", spec)
	}

	org, repo, ok := strings.Cut(repoPart, "/")
	if !ok || org == "" || repo == "" || strings.Contains(repo, "/") {
		return PullSpec{}, fmt.Errorf("invalid pull spec %q: expected org/repo:quant", spec)
	}

	return PullSpec{
		Org:   org,
		Repo:  repo,
		Quant: quant,
	}, nil
}

// MatchQuant filters a list of GGUFFiles by quant string.
// Matching is case-insensitive substring against the filename.
// Returns all matches — caller decides what to do with ambiguity.
func MatchQuant(files []GGUFFile, quant string) []GGUFFile {
	needle := strings.ToLower(quant)
	var matches []GGUFFile
	for _, file := range files {
		if strings.Contains(strings.ToLower(file.Name), needle) {
			matches = append(matches, file)
		}
	}
	return matches
}

// PullOpts configures a pull operation.
type PullOpts struct {
	ModelDir string // destination directory
	HFToken  string // optional, for gated models
	// Progress callback: called with (bytesDownloaded, totalBytes).
	// If nil, no progress reporting.
	OnProgress func(downloaded, total int64)
}

var splitGGUFPattern = regexp.MustCompile(`(?i)-\d+-of-\d+`)

// IsSplitGGUFFile reports whether a GGUF filename looks like a shard.
func IsSplitGGUFFile(name string) bool {
	return splitGGUFPattern.MatchString(name)
}

// GuessQuantFromFilename returns a best-effort quant label derived from a GGUF filename.
func GuessQuantFromFilename(name string) string {
	base := strings.TrimSuffix(name, filepath.Ext(name))
	if i := strings.LastIndex(base, "-"); i >= 0 && i < len(base)-1 {
		return base[i+1:]
	}
	return base
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

func (cw *countingWriter) Count() int64 {
	return cw.count
}

// Pull downloads a GGUF from HuggingFace.
// 1. Parses the spec
// 2. Lists GGUFs in the repo
// 3. Matches quant filter
// 4. If 0 matches: error with available quants listed
// 5. If >1 match: error with matched files listed (not ambiguous — be specific)
// 6. If exactly 1 match: download to modelDir with original filename
// 7. Resume support: if partial file exists, send Range header, append
// 8. SHA256 verification after download
func Pull(spec string, opts PullOpts) (resultPath string, err error) {
	parsed, err := ParseSpec(spec)
	if err != nil {
		return "", err
	}
	if opts.ModelDir == "" {
		return "", errors.New("model dir is required")
	}
	if err := os.MkdirAll(opts.ModelDir, 0755); err != nil {
		return "", fmt.Errorf("creating model dir %s: %w", opts.ModelDir, err)
	}

	files, err := ListGGUFs(parsed.Org, parsed.Repo, opts.HFToken)
	if err != nil {
		return "", err
	}

	matches := MatchQuant(files, parsed.Quant)
	for _, file := range matches {
		if IsSplitGGUFFile(file.Name) {
			return "", fmt.Errorf(
				"split GGUF shards are not supported yet: %s\nDownload this model manually from HuggingFace instead.",
				file.Name,
			)
		}
	}

	if len(matches) == 0 {
		return "", formatNoMatchError(parsed, files)
	}
	if len(matches) > 1 {
		return "", formatAmbiguousError(parsed, matches)
	}

	selected := matches[0]
	finalPath := filepath.Join(opts.ModelDir, selected.Name)
	partialPath := finalPath + ".partial"

	if ok, err := verifyExisting(finalPath, selected.SHA256); err != nil {
		return "", err
	} else if ok {
		return finalPath, nil
	}

	if _, err := os.Stat(partialPath); err == nil {
		if err := downloadFile(parsed, selected, partialPath, opts.HFToken, opts.OnProgress); err != nil {
			return "", err
		}
	} else if errors.Is(err, os.ErrNotExist) {
		if err := downloadFile(parsed, selected, partialPath, opts.HFToken, opts.OnProgress); err != nil {
			return "", err
		}
	} else if err != nil {
		return "", fmt.Errorf("checking existing partial file %s: %w", partialPath, err)
	}

	if selected.SHA256 != "" {
		if err := verifyFileSHA256(partialPath, selected.SHA256); err != nil {
			_ = os.Remove(partialPath)
			return "", err
		}
	}

	if err := os.Rename(partialPath, finalPath); err != nil {
		return "", fmt.Errorf("finalizing download to %s: %w", finalPath, err)
	}

	return finalPath, nil
}

func verifyExisting(path, expectedSHA string) (bool, error) {
	_, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, err
	}

	if expectedSHA == "" {
		fmt.Fprintf(os.Stderr, "Warning: existing file %s found, but HuggingFace did not provide a SHA256; skipping verification\n", path)
		return true, nil
	}

	actual, err := sha256OfFile(path)
	if err != nil {
		return false, err
	}
	if strings.EqualFold(actual, expectedSHA) {
		fmt.Fprintf(os.Stderr, "already downloaded: %s\n", path)
		return true, nil
	}

	return false, fmt.Errorf("file %s already exists but SHA256 doesn't match (got %s, expected %s) — delete it and re-run pull", path, actual, expectedSHA)
}

func downloadFile(spec PullSpec, file GGUFFile, partialPath, hfToken string, onProgress func(int64, int64)) error {
	req, err := http.NewRequest(http.MethodGet, file.DownloadURL, nil)
	if err != nil {
		return err
	}
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}

	existingBytes := int64(0)
	if info, err := os.Stat(partialPath); err == nil {
		existingBytes = info.Size()
		if existingBytes > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingBytes))
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("checking partial file %s: %w", partialPath, err)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		existingBytes = 0
	case http.StatusPartialContent:
		// Resume supported.
	case http.StatusRequestedRangeNotSatisfiable:
		if selected, err := verifyCompletedPartial(partialPath, file.SHA256); err == nil && selected {
			return nil
		}
		return fmt.Errorf("range request not satisfiable for %s and partial file could not be verified", file.Name)
	case http.StatusNotFound:
		return fmt.Errorf("HuggingFace repo %s/%s not found or private (set HF_TOKEN if this is gated)", spec.Org, spec.Repo)
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("authentication required for HuggingFace repo %s/%s (set HF_TOKEN)", spec.Org, spec.Repo)
	default:
		return fmt.Errorf("unexpected HTTP status %d while downloading %s", resp.StatusCode, file.Name)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if resp.StatusCode == http.StatusPartialContent && existingBytes > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
		existingBytes = 0
	}

	out, err := os.OpenFile(partialPath, flags, 0644)
	if err != nil {
		return fmt.Errorf("opening partial file %s: %w", partialPath, err)
	}
	defer out.Close()

	lastReported := existingBytes
	writer := &countingWriter{
		w: out,
		onChange: func(downloaded int64) {
			if onProgress != nil {
				onProgress(existingBytes+downloaded, file.Size)
			}
			lastReported = existingBytes + downloaded
		},
	}

	if existingBytes > 0 && onProgress != nil {
		onProgress(existingBytes, file.Size)
	}

	if _, err := io.Copy(writer, resp.Body); err != nil {
		return fmt.Errorf("downloading %s: %w", file.Name, err)
	}

	if onProgress != nil {
		onProgress(lastReported, file.Size)
	}

	return nil
}

func verifyCompletedPartial(path, expectedSHA string) (bool, error) {
	if _, err := os.Stat(path); err != nil {
		return false, err
	}
	if expectedSHA == "" {
		fmt.Fprintf(os.Stderr, "Warning: cannot verify %s because HuggingFace did not provide a SHA256\n", path)
		return true, nil
	}

	actual, err := sha256OfFile(path)
	if err != nil {
		return false, err
	}
	if !strings.EqualFold(actual, expectedSHA) {
		return false, fmt.Errorf("partial file %s has SHA256 %s, expected %s", path, actual, expectedSHA)
	}
	return true, nil
}

func verifyFileSHA256(path, expectedSHA string) error {
	if expectedSHA == "" {
		return nil
	}

	actual, err := sha256OfFile(path)
	if err != nil {
		return err
	}
	if !strings.EqualFold(actual, expectedSHA) {
		return fmt.Errorf("SHA256 mismatch for %s: got %s, want %s", path, actual, expectedSHA)
	}
	return nil
}

func sha256OfFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func formatNoMatchError(spec PullSpec, files []GGUFFile) error {
	var b strings.Builder
	fmt.Fprintf(&b, "No match for %q in %s/%s\n\n", spec.Quant, spec.Org, spec.Repo)
	b.WriteString("Available:\n")
	if len(files) == 0 {
		b.WriteString("  (no GGUF files found)\n")
		return errors.New(b.String())
	}
	for _, file := range files {
		fmt.Fprintf(&b, "  %-9s %-40s (%s)\n", GuessQuantFromFilename(file.Name), file.Name, humanSize(file.Size))
	}
	return errors.New(strings.TrimRight(b.String(), "\n"))
}

func formatAmbiguousError(spec PullSpec, matches []GGUFFile) error {
	var b strings.Builder
	fmt.Fprintf(&b, "Multiple matches for %q in %s/%s:\n", spec.Quant, spec.Org, spec.Repo)
	for _, file := range matches {
		fmt.Fprintf(&b, "  %-9s %-40s (%s)\n", GuessQuantFromFilename(file.Name), file.Name, humanSize(file.Size))
	}
	fmt.Fprintf(&b, "\nBe more specific: nollama pull %s/%s:%s", spec.Org, spec.Repo, GuessQuantFromFilename(matches[0].Name))
	return errors.New(strings.TrimRight(b.String(), "\n"))
}

func humanSize(bytes int64) string {
	const gb = 1024 * 1024 * 1024
	const mb = 1024 * 1024
	if bytes >= gb {
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(gb))
	}
	if bytes >= mb {
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(mb))
	}
	return fmt.Sprintf("%d B", bytes)
}
