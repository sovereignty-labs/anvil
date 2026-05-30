package main

import (
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// passthroughHarness builds a cobra command that mimics `anvil load`'s arg
// shape so we can drive collectPassthrough / exactPositionalArgs through
// cobra's actual flag parser (including ArgsLenAtDash).
func passthroughHarness(t *testing.T, cli []string) (collected []string, args []string, err error) {
	t.Helper()
	cmd := &cobra.Command{
		Use:  "fake",
		Args: exactPositionalArgs(1),
		RunE: func(c *cobra.Command, a []string) error {
			collected = collectPassthrough(c, a)
			args = a
			return nil
		},
	}
	cmd.Flags().StringArray("passthrough", nil, "")
	cmd.SetArgs(cli)
	err = cmd.Execute()
	return
}

func TestDashSeparatorCapturesPostDashArgs(t *testing.T) {
	got, _, err := passthroughHarness(t, []string{"model.gguf", "--", "--ctx-size", "32768", "--parallel", "4"})
	if err != nil {
		t.Fatalf("cmd.Execute: %v", err)
	}
	want := []string{"--ctx-size", "32768", "--parallel", "4"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestPassthroughValueSplitsOnWhitespace(t *testing.T) {
	got, _, err := passthroughHarness(t, []string{"--passthrough", "--ctx-size 32768", "--passthrough", "--parallel 4", "model.gguf"})
	if err != nil {
		t.Fatalf("cmd.Execute: %v", err)
	}
	want := []string{"--ctx-size", "32768", "--parallel", "4"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestPassthroughBooleanFlagPassesThrough(t *testing.T) {
	got, _, err := passthroughHarness(t, []string{"--passthrough", "--flash-attn", "model.gguf"})
	if err != nil {
		t.Fatalf("cmd.Execute: %v", err)
	}
	if len(got) != 1 || got[0] != "--flash-attn" {
		t.Errorf("got %v, want [--flash-attn]", got)
	}
}

func TestDashAndPassthroughProduceIdenticalResults(t *testing.T) {
	dashOut, _, err := passthroughHarness(t, []string{"model.gguf", "--", "--ctx-size", "32768", "--parallel", "4"})
	if err != nil {
		t.Fatalf("dash variant: %v", err)
	}
	ptOut, _, err := passthroughHarness(t, []string{"--passthrough", "--ctx-size 32768", "--passthrough", "--parallel 4", "model.gguf"})
	if err != nil {
		t.Fatalf("passthrough variant: %v", err)
	}
	if strings.Join(dashOut, " ") != strings.Join(ptOut, " ") {
		t.Errorf("dash=%v, passthrough=%v — should match", dashOut, ptOut)
	}
}

func TestExactPositionalArgsRejectsExtraBeforeDash(t *testing.T) {
	_, _, err := passthroughHarness(t, []string{"model.gguf", "extra-arg"})
	if err == nil {
		t.Error("expected error when too many positional args before --")
	}
}

func TestExactPositionalArgsAllowsManyAfterDash(t *testing.T) {
	_, _, err := passthroughHarness(t, []string{"model.gguf", "--", "a", "b", "c", "d"})
	if err != nil {
		t.Errorf("post-dash tokens should not count as positionals, got: %v", err)
	}
}
