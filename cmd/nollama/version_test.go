package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestVersionCommandRuns(t *testing.T) {
	var out bytes.Buffer
	versionCmd.SetOut(&out)
	t.Cleanup(func() { versionCmd.SetOut(nil) })

	if err := versionCmd.RunE(versionCmd, nil); err != nil {
		t.Fatalf("version command failed: %v", err)
	}

	got := out.String()
	if !strings.HasPrefix(got, "nollama ") {
		t.Errorf("expected output to start with 'nollama ', got %q", got)
	}
}
