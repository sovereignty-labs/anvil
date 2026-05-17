package profiles

import "embed"

// FS contains the built-in load profiles shipped with nollama.
//
//go:embed *.yaml
var FS embed.FS
