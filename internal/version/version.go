// Package version exposes build-time identity baked in via ldflags.
package version

import "fmt"

// Set via -ldflags at build time. Defaults match an unstamped local build.
var (
	Version = "dev"
	Commit  = "none"
	Date    = "unknown"
)

// String returns a human-readable build identifier suitable for `--version`
// output. Handles three cases:
//   - All three set: "nollama v0.1.0 (abc1234) built 2026-05-19T12:00:00Z"
//   - Version unset/dev, commit + date set: "nollama dev (abc1234) built ..."
//   - Nothing meaningful: "nollama dev (unknown)"
func String() string {
	v := Version
	if v == "" {
		v = "dev"
	}
	commit := Commit
	if commit == "" {
		commit = "none"
	}
	date := Date
	if date == "" {
		date = "unknown"
	}

	// "All empty" sentinel: commit + date both at defaults, version dev.
	if v == "dev" && commit == "none" && date == "unknown" {
		return "nollama dev (unknown)"
	}

	// Version-known path drops the leading "v" duplication when the tag
	// already starts with v (e.g., v0.1.0 stays v0.1.0; 0.1.0 becomes v0.1.0).
	prefix := v
	if v != "dev" && len(v) > 0 && v[0] != 'v' {
		prefix = "v" + v
	}

	if v == "dev" {
		return fmt.Sprintf("nollama dev (%s) built %s", commit, date)
	}
	return fmt.Sprintf("nollama %s (%s) built %s", prefix, commit, date)
}
