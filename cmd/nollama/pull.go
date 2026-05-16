package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/hirdforge/nollama/internal/config"
	"github.com/hirdforge/nollama/internal/pull"
	"github.com/spf13/cobra"
)

var (
	pullModelDir string
)

var pullCmd = &cobra.Command{
	Use:   "pull <org/repo:quant>",
	Short: "Pull a GGUF from HuggingFace",
	Long:  "Download a single GGUF file from HuggingFace using an explicit quant filter.",
	Args:  cobra.ExactArgs(1),
	RunE:  runPull,
}

func init() {
	pullCmd.Flags().StringVar(&pullModelDir, "model-dir", "", "Directory to save models")
}

func runPull(_ *cobra.Command, args []string) error {
	spec := args[0]
	parsed, err := pull.ParseSpec(spec)
	if err != nil {
		return fatalPull(err)
	}

	modelDir, err := resolvePullModelDir()
	if err != nil {
		return fatalPull(err)
	}
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return fatalPull(fmt.Errorf("create model dir %s: %w", modelDir, err))
	}

	token := os.Getenv("HF_TOKEN")

	fmt.Fprintf(os.Stderr, "Fetching file list from %s/%s...\n", parsed.Org, parsed.Repo)
	files, err := pull.ListGGUFs(parsed.Org, parsed.Repo, token)
	if err != nil {
		return fatalPull(err)
	}

	matches := pull.MatchQuant(files, parsed.Quant)
	for _, file := range matches {
		if pull.IsSplitGGUFFile(file.Name) {
			return fatalPull(fmt.Errorf(
				"split GGUF shards are not supported yet: %s\nDownload this model manually from HuggingFace instead.",
				file.Name,
			))
		}
	}

	switch len(matches) {
	case 0:
		printNoMatch(parsed, files)
		return fatalPull(fmt.Errorf("no matching GGUF found"))
	case 1:
		fmt.Fprintf(os.Stderr, "Found: %s (%s)\n\n", matches[0].Name, humanSize(matches[0].Size))
	default:
		printAmbiguous(parsed, matches)
		return fatalPull(fmt.Errorf("multiple matches found"))
	}

	fmt.Fprintf(os.Stderr, "Downloading to %s\n", filepath.Join(modelDir, matches[0].Name))

	start := time.Now()
	resumeBase := int64(-1)
	progress := func(downloaded, total int64) {
		if resumeBase < 0 {
			resumeBase = downloaded
			start = time.Now()
		}

		sessionBytes := downloaded - resumeBase
		elapsed := time.Since(start).Seconds()
		speed := float64(0)
		if elapsed > 0 && sessionBytes > 0 {
			speed = float64(sessionBytes) / elapsed
		}

		eta := "--"
		if speed > 0 && total > downloaded {
			seconds := float64(total-downloaded) / speed
			eta = formatDuration(seconds)
		}

		fmt.Fprintf(os.Stderr, "\r  %s / %s  %3d%%  %s/s  ETA %s",
			humanSize(downloaded),
			humanSize(total),
			percent(downloaded, total),
			humanRate(speed),
			eta,
		)
	}

	resultPath, err := pull.Pull(spec, pull.PullOpts{
		ModelDir:   modelDir,
		HFToken:    token,
		OnProgress: progress,
	})
	if err != nil {
		return fatalPull(err)
	}

	fmt.Fprintln(os.Stderr)

	info, err := os.Stat(resultPath)
	if err != nil {
		return fatalPull(fmt.Errorf("stat downloaded file %s: %w", resultPath, err))
	}

	fmt.Printf("%s (%s verified)\n", resultPath, humanSize(info.Size()))
	return nil
}

func fatalPull(err error) error {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
	return nil
}

func resolvePullModelDir() (string, error) {
	if pullModelDir != "" {
		return pullModelDir, nil
	}

	cfgPath := config.FindConfig()
	if cfgPath != "" {
		cfg, err := config.Load(cfgPath)
		if err != nil {
			return "", err
		}
		if cfg.ModelDir != "" {
			return cfg.ModelDir, nil
		}
	}

	return config.DefaultConfig().ModelDir, nil
}

func printNoMatch(spec pull.PullSpec, files []pull.GGUFFile) {
	fmt.Fprintf(os.Stderr, "No match for %q in %s/%s\n\n", spec.Quant, spec.Org, spec.Repo)
	fmt.Fprintln(os.Stderr, "Available:")
	for _, file := range files {
		fmt.Fprintf(os.Stderr, "  %-9s %-40s (%s)\n", pull.GuessQuantFromFilename(file.Name), file.Name, humanSize(file.Size))
	}
}

func printAmbiguous(spec pull.PullSpec, matches []pull.GGUFFile) {
	fmt.Fprintf(os.Stderr, "Multiple matches for %q in %s/%s:\n", spec.Quant, spec.Org, spec.Repo)
	for _, file := range matches {
		fmt.Fprintf(os.Stderr, "  %-9s %-40s (%s)\n", pull.GuessQuantFromFilename(file.Name), file.Name, humanSize(file.Size))
	}
	fmt.Fprintf(os.Stderr, "\nBe more specific: nollama pull %s/%s:%s\n", spec.Org, spec.Repo, pull.GuessQuantFromFilename(matches[0].Name))
}

func humanSize(bytes int64) string {
	const gb = 1024 * 1024 * 1024
	const mb = 1024 * 1024
	switch {
	case bytes >= gb:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(gb))
	case bytes >= mb:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(mb))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

func humanRate(rate float64) string {
	if rate <= 0 {
		return "--"
	}
	const gb = 1024.0 * 1024.0 * 1024.0
	const mb = 1024.0 * 1024.0
	switch {
	case rate >= gb:
		return fmt.Sprintf("%.1f GB", rate/gb)
	case rate >= mb:
		return fmt.Sprintf("%.0f MB", rate/mb)
	default:
		return fmt.Sprintf("%.0f B", rate)
	}
}

func percent(downloaded, total int64) int {
	if total <= 0 {
		return 0
	}
	if downloaded >= total {
		return 100
	}
	return int((downloaded * 100) / total)
}

func formatDuration(seconds float64) string {
	if seconds < 1 {
		return "<1s"
	}
	return fmt.Sprintf("%.0fs", seconds)
}
