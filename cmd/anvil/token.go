package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var tokenCmd = &cobra.Command{
	Use:   "token",
	Short: "Manage HuggingFace authentication token",
	Long: `Manage the HuggingFace token used by pull / search / run.

Stored at $XDG_CONFIG_HOME/anvil/token (or ~/.config/anvil/token) with
mode 0600. When both the file and the HF_TOKEN env var are present, the
file wins.`,
}

var tokenSetCmd = &cobra.Command{
	Use:   "set <token>",
	Short: "Store a HuggingFace token",
	Args:  cobra.ExactArgs(1),
	RunE:  runTokenSet,
}

var tokenShowCmd = &cobra.Command{
	Use:   "show",
	Short: "Show the configured token (masked) and its source",
	Args:  cobra.NoArgs,
	RunE:  runTokenShow,
}

var tokenRMCmd = &cobra.Command{
	Use:     "rm",
	Aliases: []string{"remove", "delete"},
	Short:   "Remove the stored token file",
	Args:    cobra.NoArgs,
	RunE:    runTokenRM,
}

// tokenFilePath returns the canonical location of the token file. Empty when
// the config dir can't be resolved (no HOME, no XDG override).
func tokenFilePath() string {
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		return filepath.Join(xdg, "anvil", "token")
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".config", "anvil", "token")
	}
	return ""
}

// tokenSource describes where a resolved HF token came from. Empty when no
// token is configured.
type tokenSource struct {
	value  string
	origin string // human-readable: file path or "HF_TOKEN env"
}

// resolveHFTokenSource returns the token and where it came from. Priority:
// stored file → HF_TOKEN env var → empty.
func resolveHFTokenSource() tokenSource {
	if path := tokenFilePath(); path != "" {
		if data, err := os.ReadFile(path); err == nil {
			if token := strings.TrimSpace(string(data)); token != "" {
				return tokenSource{value: token, origin: path}
			}
		}
	}
	if env := strings.TrimSpace(os.Getenv("HF_TOKEN")); env != "" {
		return tokenSource{value: env, origin: "HF_TOKEN env"}
	}
	return tokenSource{}
}

// resolveHFToken is a convenience wrapper returning just the token string.
// Other CLI commands (pull, search, run) call this instead of reading
// HF_TOKEN directly.
func resolveHFToken() string {
	return resolveHFTokenSource().value
}

// maskToken returns a display form like "hf_NChK...sDSti" so an over-the-
// shoulder reader can't copy the secret out of anvil token show.
func maskToken(token string) string {
	if len(token) <= 11 {
		return strings.Repeat("*", len(token))
	}
	return token[:6] + "..." + token[len(token)-5:]
}

func runTokenSet(_ *cobra.Command, args []string) error {
	token := strings.TrimSpace(args[0])
	if token == "" {
		return fmt.Errorf("token cannot be empty")
	}
	path := tokenFilePath()
	if path == "" {
		return fmt.Errorf("cannot determine config directory (HOME not set)")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create config dir: %w", err)
	}
	if err := os.WriteFile(path, []byte(token+"\n"), 0o600); err != nil {
		return fmt.Errorf("write token: %w", err)
	}
	fmt.Fprintf(os.Stderr, "Token saved to %s\n", path)
	return nil
}

func runTokenShow(_ *cobra.Command, _ []string) error {
	src := resolveHFTokenSource()
	if src.value == "" {
		fmt.Fprintln(os.Stderr, "No HuggingFace token configured.")
		fmt.Fprintln(os.Stderr, "Set one with: anvil token set <token>")
		fmt.Fprintln(os.Stderr, "Or export HF_TOKEN=<token>")
		return nil
	}
	fmt.Printf("Source: %s\n", src.origin)
	fmt.Printf("Token:  %s\n", maskToken(src.value))
	return nil
}

func runTokenRM(_ *cobra.Command, _ []string) error {
	path := tokenFilePath()
	if path == "" {
		return fmt.Errorf("cannot determine config directory (HOME not set)")
	}
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			fmt.Fprintln(os.Stderr, "No stored token to remove.")
			return nil
		}
		return fmt.Errorf("remove token: %w", err)
	}
	fmt.Fprintf(os.Stderr, "Removed %s\n", path)
	return nil
}
