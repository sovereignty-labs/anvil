package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"
)

var searchCmd = &cobra.Command{
	Use:   "search <query...>",
	Short: "Search HuggingFace for GGUF model repos",
	Long: `Search HuggingFace for GGUF model repositories matching a free-text query.

Results are sorted by download count (most popular first), and quantization
variants are listed for each repo.

Examples:
  nollama search gemma 4 e2b
  nollama search qwen 4b`,
	Args: cobra.MinimumNArgs(1),
	RunE: runSearch,
}

// searchEndpoint is the HuggingFace models index. Overridden in tests so we
// don't hit the public API during go test ./....
var searchEndpoint = "https://huggingface.co/api/models"

// quantPattern matches GGUF quant suffixes in filenames (Q4_K_M, IQ4_XS,
// BF16, etc). Mirrors the regex in internal/model/gguf.go.
var quantPattern = regexp.MustCompile(`(?i)[_-]((?:IQ|Q)\d+(?:_K)?(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4)(?:[_.-]|$)`)

// hfModel is the subset of the HuggingFace API model object we display.
type hfModel struct {
	ModelID   string   `json:"modelId"`
	Downloads int64    `json:"downloads"`
	Tags      []string `json:"tags"`
	Siblings  []struct {
		Filename string `json:"rfilename"`
		Size     int64  `json:"size,omitempty"`
	} `json:"siblings"`
}

func runSearch(_ *cobra.Command, args []string) error {
	query := strings.TrimSpace(strings.Join(args, " "))
	if query == "" {
		return fmt.Errorf("search query cannot be empty")
	}

	results, err := searchHuggingFace(query, resolveHFToken())
	if err != nil {
		return err
	}
	if len(results) == 0 {
		fmt.Fprintf(os.Stderr, "No GGUF repos matched %q.\n", query)
		return nil
	}

	const limit = 10
	if len(results) > limit {
		results = results[:limit]
	}

	renderSearchResults(os.Stdout, results)
	return nil
}

// searchHuggingFace calls the HF models index for `<query> GGUF` and returns
// repos sorted by downloads descending. Empty token sends an unauthenticated
// request (works for public repos).
func searchHuggingFace(query, token string) ([]hfModel, error) {
	params := url.Values{}
	params.Set("search", query+" GGUF")
	params.Set("filter", "gguf")
	params.Set("sort", "downloads")
	params.Set("direction", "-1")
	params.Set("limit", "20")
	params.Set("full", "true") // requests siblings on the same response

	req, err := http.NewRequest(http.MethodGet, searchEndpoint+"?"+params.Encode(), nil)
	if err != nil {
		return nil, fmt.Errorf("build search request: %w", err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call HuggingFace API: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("HuggingFace search returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	var results []hfModel
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return nil, fmt.Errorf("decode HuggingFace response: %w", err)
	}
	// API claims to sort but be defensive — some responses come back unsorted.
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].Downloads > results[j].Downloads
	})
	return results, nil
}

// renderSearchResults writes a tab-formatted table of repos to w with a
// "Pull with:" hint at the end.
func renderSearchResults(w io.Writer, results []hfModel) {
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "REPO\tDOWNLOADS\tQUANTS")
	for _, m := range results {
		quants := extractQuants(m.Siblings)
		quantStr := "—"
		if len(quants) > 0 {
			quantStr = strings.Join(quants, ", ")
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", m.ModelID, formatThousands(m.Downloads), quantStr)
	}
	_ = tw.Flush()
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Pull with: nollama pull <repo>:<quant>")
}

// extractQuants scans GGUF filenames and returns a sorted, deduplicated list
// of the embedded quant identifiers.
func extractQuants(siblings []struct {
	Filename string `json:"rfilename"`
	Size     int64  `json:"size,omitempty"`
}) []string {
	seen := make(map[string]struct{})
	for _, s := range siblings {
		if !strings.HasSuffix(strings.ToLower(s.Filename), ".gguf") {
			continue
		}
		matches := quantPattern.FindAllStringSubmatch(strings.ToUpper(s.Filename), -1)
		for _, m := range matches {
			if len(m) < 2 {
				continue
			}
			seen[m[1]] = struct{}{}
		}
	}
	quants := make([]string, 0, len(seen))
	for q := range seen {
		quants = append(quants, q)
	}
	sort.Strings(quants)
	return quants
}

// formatThousands renders 12345 as "12,345". Negative/small values pass
// through unchanged (downloads should never be < 1000 in practice but we
// don't blow up on weird inputs).
func formatThousands(n int64) string {
	if n < 0 {
		return fmt.Sprintf("%d", n)
	}
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	pre := len(s) % 3
	if pre > 0 {
		b.WriteString(s[:pre])
		if len(s) > pre {
			b.WriteByte(',')
		}
	}
	for i := pre; i < len(s); i += 3 {
		b.WriteString(s[i : i+3])
		if i+3 < len(s) {
			b.WriteByte(',')
		}
	}
	return b.String()
}
