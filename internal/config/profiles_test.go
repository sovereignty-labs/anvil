package config

import "testing"

func TestLoadBuiltinProfiles(t *testing.T) {
	profiles, err := LoadBuiltinProfiles()
	if err != nil {
		t.Fatalf("LoadBuiltinProfiles failed: %v", err)
	}

	if len(profiles) != 3 {
		t.Fatalf("expected 3 builtin profiles, got %d", len(profiles))
	}

	profile, ok := profiles["turboquant-asymmetric"]
	if !ok {
		t.Fatalf("expected turboquant-asymmetric profile")
	}
	if profile.Requires.Runtime != "turboquant" {
		t.Fatalf("expected runtime requirement turboquant, got %q", profile.Requires.Runtime)
	}
	if profile.Flags["cache-type-v"] != "turbo3" {
		t.Fatalf("expected cache-type-v turbo3, got %#v", profile.Flags["cache-type-v"])
	}
}

func TestMergeProfilesOverrideSemantics(t *testing.T) {
	first := Profile{
		Name: "first",
		Flags: map[string]interface{}{
			"ctx-size":   32768,
			"flash-attn": "on",
		},
		Requires: ProfileRequires{Runtime: "turboquant"},
	}
	second := Profile{
		Name: "second",
		Flags: map[string]interface{}{
			"ctx-size": 65536,
			"parallel": 8,
		},
	}

	merged := MergeProfiles([]Profile{first, second})
	if merged.Flags["ctx-size"] != 65536 {
		t.Fatalf("expected later profile to override ctx-size, got %#v", merged.Flags["ctx-size"])
	}
	if merged.Flags["parallel"] != 8 {
		t.Fatalf("expected merged parallel flag, got %#v", merged.Flags["parallel"])
	}
	if len(merged.Requires) != 1 || merged.Requires[0].Runtime != "turboquant" {
		t.Fatalf("expected merged runtime requirement, got %#v", merged.Requires)
	}
}

func TestProfileRuntimeWarnings(t *testing.T) {
	warnings := ProfileRuntimeWarnings([]ProfileRequires{{Runtime: "turboquant"}}, "llama-mainline")
	if len(warnings) != 1 {
		t.Fatalf("expected 1 warning, got %d", len(warnings))
	}
	if warnings[0] == "" {
		t.Fatal("expected non-empty warning")
	}

	warnings = ProfileRuntimeWarnings([]ProfileRequires{{Runtime: "turboquant"}}, "turboquant")
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings on runtime match, got %#v", warnings)
	}
}

func TestMergedFlagsWithProfiles(t *testing.T) {
	cfg := &Config{
		Defaults: map[string]interface{}{
			"flash-attn": "off",
			"parallel":   2,
		},
	}
	entry := AutoloadEntry{
		Profiles: []string{"agent-fleet"},
		Flags: map[string]interface{}{
			"ctx-size": 131072,
		},
	}

	merged, requires, err := cfg.MergedFlagsWithProfiles(entry)
	if err != nil {
		t.Fatalf("MergedFlagsWithProfiles failed: %v", err)
	}

	if merged["parallel"] != 8 {
		t.Fatalf("expected profile to override default parallel, got %#v", merged["parallel"])
	}
	if merged["ctx-size"] != 131072 {
		t.Fatalf("expected entry flag to override profile ctx-size, got %#v", merged["ctx-size"])
	}
	if len(requires) != 0 {
		t.Fatalf("expected no runtime requirements, got %#v", requires)
	}
}
