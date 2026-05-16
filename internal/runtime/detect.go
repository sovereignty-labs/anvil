package runtime

import (
	"fmt"
	osexec "os/exec"
	stdlib "runtime"
	"sort"
	"strconv"
	"strings"
)

// Platform describes the current OS/arch/GPU capability.
type Platform struct {
	OS   string // "linux", "darwin", "windows"
	Arch string // "amd64", "arm64"
	CUDA string // "available" when NVIDIA CUDA is available, otherwise empty
}

var (
	lookPath = osexec.LookPath
	command  = osexec.Command

	execLookPath = lookPath
	execCommand  = command
)

// DetectPlatform detects the local platform and whether CUDA is available.
func DetectPlatform() Platform {
	p := Platform{
		OS:   stdlib.GOOS,
		Arch: stdlib.GOARCH,
	}

	if path, err := execLookPath("nvidia-smi"); err == nil && path != "" {
		cmd := execCommand(path)
		if err := cmd.Run(); err == nil {
			p.CUDA = "available"
		}
	}

	return p
}

// SelectAsset chooses the best archive for the current platform.
func SelectAsset(assets []ReleaseAsset, platform Platform) (*ReleaseAsset, error) {
	candidates := make([]scoredAsset, 0, len(assets))
	for _, asset := range assets {
		score, ok := scoreAsset(asset, platform)
		if !ok {
			continue
		}
		candidates = append(candidates, scoredAsset{asset: asset, score: score})
	}

	if len(candidates) == 0 {
		names := make([]string, 0, len(assets))
		for _, asset := range assets {
			names = append(names, asset.Name)
		}
		return nil, fmt.Errorf("no llama.cpp release asset matched platform %s/%s (CUDA=%t); available: %s",
			platform.OS, platform.Arch, platform.CUDA != "", strings.Join(names, ", "))
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		if candidates[i].asset.Size != candidates[j].asset.Size {
			return candidates[i].asset.Size > candidates[j].asset.Size
		}
		return candidates[i].asset.Name < candidates[j].asset.Name
	})

	selected := candidates[0].asset
	return &selected, nil
}

type scoredAsset struct {
	asset ReleaseAsset
	score int
}

func scoreAsset(asset ReleaseAsset, platform Platform) (int, bool) {
	name := strings.ToLower(asset.Name)
	if !hasArchiveSuffix(name) {
		return 0, false
	}

	if !matchesPlatform(name, platform.OS) {
		return 0, false
	}
	if !matchesArch(name, platform.Arch) {
		return 0, false
	}

	cuda := strings.Contains(name, "cuda")
	score := 0
	if platform.CUDA != "" {
		if cuda {
			score += 200
		} else {
			score += 100
		}
	} else {
		if cuda {
			score += 20
		} else {
			score += 200
		}
	}

	if cuda {
		score += cudaVersionScore(name)
	}

	return score, true
}

func hasArchiveSuffix(name string) bool {
	return strings.HasSuffix(name, ".zip") || strings.HasSuffix(name, ".tar.gz")
}

func matchesPlatform(name, os string) bool {
	switch os {
	case "linux":
		return strings.Contains(name, "ubuntu") || strings.Contains(name, "linux")
	case "darwin":
		return strings.Contains(name, "macos")
	case "windows":
		return strings.Contains(name, "win")
	default:
		return false
	}
}

func matchesArch(name, arch string) bool {
	switch arch {
	case "amd64":
		return strings.Contains(name, "x64") || strings.Contains(name, "amd64") || strings.Contains(name, "x86_64")
	case "arm64":
		return strings.Contains(name, "arm64") || strings.Contains(name, "aarch64")
	default:
		return false
	}
}

func cudaVersionScore(name string) int {
	idx := strings.Index(name, "cu")
	if idx < 0 || idx+2 >= len(name) {
		return 0
	}

	rest := name[idx+2:]
	rest = strings.TrimLeft(rest, "-_")

	var parts []int
	token := strings.Builder{}
	for _, r := range rest {
		switch {
		case r >= '0' && r <= '9':
			token.WriteRune(r)
		case r == '.':
			if token.Len() > 0 {
				n, _ := strconv.Atoi(token.String())
				parts = append(parts, n)
				token.Reset()
			}
		default:
			if token.Len() > 0 {
				n, _ := strconv.Atoi(token.String())
				parts = append(parts, n)
				token.Reset()
			}
			goto done
		}
	}
done:
	if token.Len() > 0 {
		n, _ := strconv.Atoi(token.String())
		parts = append(parts, n)
	}

	score := 0
	for i, part := range parts {
		score += part
		if i == 0 {
			score *= 1000
		}
	}
	return score
}
