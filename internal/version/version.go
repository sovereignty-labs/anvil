package version

import "fmt"

// Set via -ldflags at build time.
var (
	Version = "dev"
	Commit  = "none"
	Date    = "unknown"
)

func String() string {
	return fmt.Sprintf("nollama %s (commit %s, built %s)", Version, Commit, Date)
}