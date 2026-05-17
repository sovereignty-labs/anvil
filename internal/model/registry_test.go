package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFuzzyMatchExact(t *testing.T) {
	available := []string{
		"gemma-4-26B-A4B-Q3_K_XL",
		"qwen3.6-27B-IQ4_XS",
	}

	got := FuzzyMatchModel("gemma-4-26B-A4B-Q3_K_XL", available)
	if got != "gemma-4-26B-A4B-Q3_K_XL" {
		t.Errorf("exact match failed: got %q", got)
	}
}

func TestFuzzyMatchStripsGGUF(t *testing.T) {
	available := []string{
		"gemma-4-26B-A4B-Q3_K_XL",
	}

	got := FuzzyMatchModel("gemma-4-26B-A4B-Q3_K_XL.gguf", available)
	if got != "gemma-4-26B-A4B-Q3_K_XL" {
		t.Errorf("expected match after stripping .gguf, got %q", got)
	}
}

func TestFuzzyMatchCaseInsensitive(t *testing.T) {
	available := []string{
		"Gemma-4-26B-A4B-Q3_K_XL",
	}

	got := FuzzyMatchModel("gemma-4-26b-a4b-q3_k_xl", available)
	if got != "Gemma-4-26B-A4B-Q3_K_XL" {
		t.Errorf("case-insensitive match failed: got %q", got)
	}
}

func TestFuzzyMatchPrefix(t *testing.T) {
	available := []string{
		"gemma-4-26B-A4B-Q3_K_XL",
		"qwen3.6-27B-IQ4_XS",
	}

	got := FuzzyMatchModel("gemma-4-26B", available)
	if got != "gemma-4-26B-A4B-Q3_K_XL" {
		t.Errorf("prefix match failed: got %q", got)
	}
}

func TestFuzzyMatchContains(t *testing.T) {
	available := []string{
		"gemma-4-26B-A4B-Q3_K_XL",
		"qwen3.6-27B-IQ4_XS",
	}

	got := FuzzyMatchModel("Q3_K_XL", available)
	if got != "gemma-4-26B-A4B-Q3_K_XL" {
		t.Errorf("contains match failed: got %q", got)
	}
}

func TestFuzzyMatchNotFound(t *testing.T) {
	available := []string{
		"gemma-4-26B-A4B-Q3_K_XL",
		"qwen3.6-27B-IQ4_XS",
	}

	got := FuzzyMatchModel("llama-3.1-70B", available)
	if got != "" {
		t.Errorf("expected empty for no match, got %q", got)
	}
}

func TestScanDirEmpty(t *testing.T) {
	dir := t.TempDir()

	models, err := ScanDir(dir)
	if err != nil {
		t.Fatalf("ScanDir failed: %v", err)
	}
	if len(models) != 0 {
		t.Errorf("expected 0 models, got %d", len(models))
	}
}

func TestScanDirFindsGGUFs(t *testing.T) {
	dir := t.TempDir()

	// Create fake GGUF files (won't have valid headers, but ScanDir handles that gracefully)
	for _, name := range []string{"model-a.gguf", "model-b.gguf", "not-a-model.txt"} {
		path := filepath.Join(dir, name)
		os.WriteFile(path, []byte("fake"), 0644)
	}

	models, err := ScanDir(dir)
	if err != nil {
		t.Fatalf("ScanDir failed: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 GGUFs, got %d", len(models))
	}

	// Should be sorted
	if models[0].Filename != "model-a.gguf" {
		t.Errorf("expected model-a.gguf first, got %s", models[0].Filename)
	}
	if models[1].Filename != "model-b.gguf" {
		t.Errorf("expected model-b.gguf second, got %s", models[1].Filename)
	}

	// Meta should be nil for fake GGUFs (ParseGGUF will fail)
	if models[0].Meta != nil {
		t.Error("expected nil meta for fake GGUF")
	}
}

func TestScanDirNonExistent(t *testing.T) {
	_, err := ScanDir("/nonexistent/directory")
	if err == nil {
		t.Error("expected error for non-existent directory")
	}
}

func TestScanDirIgnoresSubdirs(t *testing.T) {
	dir := t.TempDir()

	// Create a subdirectory with a GGUF in it
	subdir := filepath.Join(dir, "subdir")
	os.MkdirAll(subdir, 0755)
	os.WriteFile(filepath.Join(subdir, "nested.gguf"), []byte("fake"), 0644)

	// Create a top-level GGUF
	os.WriteFile(filepath.Join(dir, "top.gguf"), []byte("fake"), 0644)

	models, err := ScanDir(dir)
	if err != nil {
		t.Fatalf("ScanDir failed: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("expected 1 model (ignoring subdirs), got %d", len(models))
	}
	if models[0].Filename != "top.gguf" {
		t.Errorf("expected top.gguf, got %s", models[0].Filename)
	}
}

func TestScanDirCacheHit(t *testing.T) {
	ClearMetadataCache()
	dir := t.TempDir()

	for _, name := range []string{"model-a.gguf", "model-b.gguf"} {
		os.WriteFile(filepath.Join(dir, name), []byte("fake"), 0644)
	}

	models1, err := ScanDir(dir)
	if err != nil {
		t.Fatalf("first ScanDir: %v", err)
	}
	models2, err := ScanDir(dir)
	if err != nil {
		t.Fatalf("second ScanDir: %v", err)
	}

	if len(models1) != len(models2) {
		t.Errorf("got different counts: %d vs %d", len(models1), len(models2))
	}
}

func TestClearMetadataCache(t *testing.T) {
	ClearMetadataCache()
	ClearMetadataCache()
}

func TestModelInfoSizeHuman(t *testing.T) {
	tests := []struct {
		bytes int64
		want  string
	}{
		{1024 * 1024 * 1024 * 12, "12.0 GB"},     // 12 GB
		{1024*1024*1024*17 + 1024*1024*512, "17.5 GB"}, // 17.5 GB
		{500 * 1024 * 1024, "500 MB"},              // 500 MB
	}

	for _, tt := range tests {
		mi := ModelInfo{SizeBytes: tt.bytes}
		got := mi.SizeHuman()
		if got != tt.want {
			t.Errorf("SizeHuman(%d) = %s, want %s", tt.bytes, got, tt.want)
		}
	}
}
