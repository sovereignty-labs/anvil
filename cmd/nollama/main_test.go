package main

import (
	"reflect"
	"testing"

	"github.com/spf13/cobra"
)

func TestResolveLoadOverlayWithoutProfiles(t *testing.T) {
	cmd := newLoadTestCommand(t)

	flags, profiles, warnings, err := resolveLoadOverlay(cmd)
	if err != nil {
		t.Fatalf("resolveLoadOverlay failed: %v", err)
	}
	if len(flags) != 0 {
		t.Fatalf("expected no overlay flags without profiles, got %#v", flags)
	}
	if len(profiles) != 0 {
		t.Fatalf("expected no applied profiles, got %#v", profiles)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no runtime warnings, got %#v", warnings)
	}
}

func TestResolveLoadOverlayWithProfilesReturnsOnlyProfileFlags(t *testing.T) {
	cmd := newLoadTestCommand(t)
	if err := cmd.Flags().Set("profile", "agent-fleet"); err != nil {
		t.Fatalf("set profile: %v", err)
	}

	flags, profiles, warnings, err := resolveLoadOverlay(cmd)
	if err != nil {
		t.Fatalf("resolveLoadOverlay failed: %v", err)
	}

	if !reflect.DeepEqual(profiles, []string{"agent-fleet"}) {
		t.Fatalf("unexpected applied profiles: %#v", profiles)
	}
	if got := flags["parallel"]; got != 8 {
		t.Fatalf("expected profile parallel=8, got %#v", got)
	}
	if _, ok := flags["jinja"]; ok {
		t.Fatalf("did not expect config default flags in overlay, got %#v", flags)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings for agent-fleet, got %#v", warnings)
	}
}

func TestBuildRemoteLoadRequestWithoutProfilesOmitsFlags(t *testing.T) {
	cmd := newLoadTestCommand(t)

	req, profiles, warnings, err := buildRemoteLoadRequest(cmd, "/models/demo.gguf")
	if err != nil {
		t.Fatalf("buildRemoteLoadRequest failed: %v", err)
	}

	if req.Flags == nil || len(req.Flags) != 0 {
		t.Fatalf("expected empty remote flags without profiles, got %#v", req.Flags)
	}
	if !reflect.DeepEqual(profiles, []string(nil)) && len(profiles) != 0 {
		t.Fatalf("expected no profiles, got %#v", profiles)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %#v", warnings)
	}
}

func newLoadTestCommand(t *testing.T) *cobra.Command {
	t.Helper()

	cmd := &cobra.Command{Use: "load"}
	cmd.Flags().StringSlice("profile", nil, "")
	cmd.Flags().Int("gpu", -1, "")
	cmd.Flags().Bool("cpu", false, "")
	return cmd
}
