package version

import (
	"strings"
	"testing"
)

func TestVersionStringFull(t *testing.T) {
	orig := saveVars()
	defer orig.restore()

	Version = "0.1.0"
	Commit = "abc1234"
	Date = "2026-05-19T12:00:00Z"

	got := String()
	want := "nollama v0.1.0 (abc1234) built 2026-05-19T12:00:00Z"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestVersionStringTagAlreadyVPrefixed(t *testing.T) {
	orig := saveVars()
	defer orig.restore()

	Version = "v1.2.3"
	Commit = "abc1234"
	Date = "2026-05-19T12:00:00Z"

	got := String()
	if !strings.Contains(got, "v1.2.3") || strings.Contains(got, "vv1.2.3") {
		t.Errorf("expected v1.2.3 (no double-v), got %q", got)
	}
}

func TestVersionStringDevWithCommitAndDate(t *testing.T) {
	orig := saveVars()
	defer orig.restore()

	Version = "dev"
	Commit = "abc1234"
	Date = "2026-05-19T12:00:00Z"

	got := String()
	want := "nollama dev (abc1234) built 2026-05-19T12:00:00Z"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestVersionStringAllEmpty(t *testing.T) {
	orig := saveVars()
	defer orig.restore()

	// "All empty" in practice means defaults — unstamped build.
	Version = "dev"
	Commit = "none"
	Date = "unknown"

	got := String()
	want := "nollama dev (unknown)"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestVersionStringEmptyStringsFallToDefaults(t *testing.T) {
	orig := saveVars()
	defer orig.restore()

	Version = ""
	Commit = ""
	Date = ""

	got := String()
	want := "nollama dev (unknown)"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

type vars struct{ v, c, d string }

func saveVars() vars                  { return vars{Version, Commit, Date} }
func (s vars) restore()               { Version, Commit, Date = s.v, s.c, s.d }
