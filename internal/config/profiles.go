package config

import (
	"fmt"
	"io/fs"
	"sort"
	"strings"

	builtinprofiles "github.com/sovereignty-labs/anvil/profiles"
	"gopkg.in/yaml.v3"
)

type ProfileRequires struct {
	Runtime string `yaml:"runtime"`
}

type Profile struct {
	Name        string                 `yaml:"name"`
	Description string                 `yaml:"description"`
	Requires    ProfileRequires        `yaml:"requires"`
	Flags       map[string]interface{} `yaml:"flags"`
}

type MergedProfile struct {
	Flags    map[string]interface{}
	Requires []ProfileRequires
}

func LoadBuiltinProfiles() (map[string]Profile, error) {
	entries, err := fs.ReadDir(builtinprofiles.FS, ".")
	if err != nil {
		return nil, fmt.Errorf("read builtin profiles: %w", err)
	}

	out := make(map[string]Profile, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}

		data, err := fs.ReadFile(builtinprofiles.FS, entry.Name())
		if err != nil {
			return nil, fmt.Errorf("read builtin profile %s: %w", entry.Name(), err)
		}

		var profile Profile
		if err := yaml.Unmarshal(data, &profile); err != nil {
			return nil, fmt.Errorf("parse builtin profile %s: %w", entry.Name(), err)
		}
		if strings.TrimSpace(profile.Name) == "" {
			return nil, fmt.Errorf("builtin profile %s is missing name", entry.Name())
		}
		if profile.Flags == nil {
			profile.Flags = map[string]interface{}{}
		}
		out[profile.Name] = profile
	}

	return out, nil
}

func LoadProfile(name string) (Profile, error) {
	profiles, err := LoadBuiltinProfiles()
	if err != nil {
		return Profile{}, err
	}

	profile, ok := profiles[name]
	if !ok {
		names := make([]string, 0, len(profiles))
		for profileName := range profiles {
			names = append(names, profileName)
		}
		sort.Strings(names)
		return Profile{}, fmt.Errorf("unknown profile %q (available: %s)", name, strings.Join(names, ", "))
	}

	return profile, nil
}

func MergeProfiles(profiles []Profile) MergedProfile {
	merged := MergedProfile{
		Flags:    map[string]interface{}{},
		Requires: []ProfileRequires{},
	}

	for _, profile := range profiles {
		for k, v := range profile.Flags {
			merged.Flags[k] = v
		}
		if strings.TrimSpace(profile.Requires.Runtime) != "" {
			merged.Requires = append(merged.Requires, profile.Requires)
		}
	}

	return merged
}

func ProfileRuntimeWarnings(requires []ProfileRequires, activeRuntime string) []string {
	activeRuntime = strings.TrimSpace(activeRuntime)
	if len(requires) == 0 {
		return nil
	}

	warnings := make([]string, 0, len(requires))
	seen := make(map[string]bool, len(requires))
	for _, req := range requires {
		runtimeName := strings.TrimSpace(req.Runtime)
		if runtimeName == "" || seen[runtimeName] {
			continue
		}
		seen[runtimeName] = true

		switch {
		case activeRuntime == "":
			warnings = append(warnings, fmt.Sprintf("profile requires runtime %q but no active runtime is configured", runtimeName))
		case activeRuntime != runtimeName:
			warnings = append(warnings, fmt.Sprintf("profile requires runtime %q but active runtime is %q", runtimeName, activeRuntime))
		}
	}

	return warnings
}
