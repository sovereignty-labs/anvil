package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestExtractQuants(t *testing.T) {
	siblings := []struct {
		Filename string `json:"rfilename"`
		Size     int64  `json:"size,omitempty"`
	}{
		{Filename: "model-Q4_K_M.gguf"},
		{Filename: "model-Q8_0.gguf"},
		{Filename: "model-IQ4_XS.gguf"},
		{Filename: "model-BF16.gguf"},
		{Filename: "README.md"},        // ignored
		{Filename: "model-Q4_K_M.gguf"}, // dedup
	}
	got := extractQuants(siblings)
	want := []string{"BF16", "IQ4_XS", "Q4_K_M", "Q8_0"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestFormatThousands(t *testing.T) {
	cases := map[int64]string{
		0:        "0",
		42:       "42",
		1234:     "1,234",
		12345:    "12,345",
		1234567:  "1,234,567",
	}
	for in, want := range cases {
		if got := formatThousands(in); got != want {
			t.Errorf("formatThousands(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestSearchHuggingFaceAgainstFakeAPI(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("search"); !strings.Contains(got, "gemma") {
			t.Errorf("expected search to include gemma, got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode([]hfModel{
			{
				ModelID:   "bartowski/google_gemma-4-E2B-it-GGUF",
				Downloads: 12345,
				Siblings: []struct {
					Filename string `json:"rfilename"`
					Size     int64  `json:"size,omitempty"`
				}{
					{Filename: "google_gemma-4-E2B-it-Q4_K_M.gguf"},
					{Filename: "google_gemma-4-E2B-it-Q8_0.gguf"},
				},
			},
			{
				ModelID:   "unsloth/google_gemma-4-E2B-it-GGUF",
				Downloads: 8901,
				Siblings: []struct {
					Filename string `json:"rfilename"`
					Size     int64  `json:"size,omitempty"`
				}{
					{Filename: "google_gemma-4-E2B-it-Q4_K_M.gguf"},
				},
			},
		})
	}))
	defer srv.Close()

	origEndpoint := searchEndpoint
	searchEndpoint = srv.URL
	t.Cleanup(func() { searchEndpoint = origEndpoint })

	results, err := searchHuggingFace("gemma 4 e2b", "")
	if err != nil {
		t.Fatalf("searchHuggingFace: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	// Downloads-descending order.
	if results[0].Downloads < results[1].Downloads {
		t.Errorf("results not sorted by downloads desc: %v", results)
	}
}

func TestSearchHuggingFaceSendsAuthHeader(t *testing.T) {
	var seenAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("[]"))
	}))
	defer srv.Close()

	origEndpoint := searchEndpoint
	searchEndpoint = srv.URL
	t.Cleanup(func() { searchEndpoint = origEndpoint })

	if _, err := searchHuggingFace("foo", "secret-token"); err != nil {
		t.Fatal(err)
	}
	if seenAuth != "Bearer secret-token" {
		t.Errorf("expected Bearer secret-token, got %q", seenAuth)
	}
}

func TestRenderSearchResultsContainsQuantsAndPullHint(t *testing.T) {
	results := []hfModel{
		{
			ModelID:   "bartowski/test-GGUF",
			Downloads: 1234,
			Siblings: []struct {
				Filename string `json:"rfilename"`
				Size     int64  `json:"size,omitempty"`
			}{
				{Filename: "test-Q4_K_M.gguf"},
			},
		},
	}
	var buf bytes.Buffer
	renderSearchResults(&buf, results)
	got := buf.String()
	if !strings.Contains(got, "bartowski/test-GGUF") {
		t.Errorf("missing modelId, got %q", got)
	}
	if !strings.Contains(got, "Q4_K_M") {
		t.Errorf("missing quant, got %q", got)
	}
	if !strings.Contains(got, "1,234") {
		t.Errorf("downloads not formatted with commas, got %q", got)
	}
	if !strings.Contains(got, "nollama pull") {
		t.Errorf("missing pull hint, got %q", got)
	}
}
