package profiles

import "embed"

// FS contains the built-in load profiles shipped with anvil.
//
//go:embed *.yaml
var FS embed.FS
