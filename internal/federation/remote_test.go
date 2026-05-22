package federation

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/sovereignty-labs/nollama/internal/config"
)

func TestAddRemove(t *testing.T) {
	reg := &RemoteRegistry{}

	if err := reg.Add("gpu-host", "http://gpu-host.example.internal:11434"); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if _, ok := reg.List()["gpu-host"]; !ok {
		t.Fatal("expected remote in registry after add")
	}
	if err := reg.Remove("gpu-host"); err != nil {
		t.Fatalf("Remove failed: %v", err)
	}
	if _, ok := reg.List()["gpu-host"]; ok {
		t.Fatal("expected remote removed from registry")
	}
}

func TestAddDuplicate(t *testing.T) {
	reg := &RemoteRegistry{}

	if err := reg.Add("gpu-host", "http://gpu-host.example.internal:11434"); err != nil {
		t.Fatalf("first Add failed: %v", err)
	}
	if err := reg.Add("gpu-host", "http://odin2.example.internal:11434"); err == nil {
		t.Fatal("expected duplicate add to fail")
	}
}

func TestRemoveNonexistent(t *testing.T) {
	reg := &RemoteRegistry{}

	if err := reg.Remove("missing"); err == nil {
		t.Fatal("expected removing missing remote to fail")
	}
}

func TestSaveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "remotes.yaml")

	reg := &RemoteRegistry{}
	if err := reg.Add("gpu-host", "http://gpu-host.example.internal:11434"); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if err := reg.Add("inference-host", "http://inference-host.example.internal:11434"); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if err := reg.Save(path); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	loaded, err := LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry failed: %v", err)
	}

	if len(loaded.Remotes) != 2 {
		t.Fatalf("expected 2 remotes after load, got %d", len(loaded.Remotes))
	}
	if loaded.Remotes["gpu-host"].URL != "http://gpu-host.example.internal:11434" {
		t.Fatalf("unexpected gpu-host url: %s", loaded.Remotes["gpu-host"].URL)
	}
	if loaded.Remotes["inference-host"].URL != "http://inference-host.example.internal:11434" {
		t.Fatalf("unexpected inference-host url: %s", loaded.Remotes["inference-host"].URL)
	}
}

func TestURLNormalization(t *testing.T) {
	reg := &RemoteRegistry{}

	if err := reg.Add("gpu-host", "http://gpu-host.example.internal:11434/"); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if got := reg.Remotes["gpu-host"].URL; got != "http://gpu-host.example.internal:11434" {
		t.Fatalf("expected trailing slash to be stripped, got %q", got)
	}
}

func TestMergeRemotes(t *testing.T) {
	reg := &RemoteRegistry{}
	if err := reg.Add("gpu-host", "http://cli-gpu-host.example.internal:11434"); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if err := reg.Add("node-3", "http://node-3.example.internal:11434"); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	cfgRemotes := map[string]config.Remote{
		"gpu-host": {URL: "http://cfg-gpu-host.example.internal:11434"},
		"inference-host": {URL: "http://inference-host.example.internal:11434"},
	}

	merged := MergeRemotes(reg, cfgRemotes)

	if got := merged["gpu-host"]; got != "http://cfg-gpu-host.example.internal:11434" {
		t.Fatalf("expected config gpu-host to win, got %q", got)
	}
	if got := merged["inference-host"]; got != "http://inference-host.example.internal:11434" {
		t.Fatalf("expected config inference-host to be present, got %q", got)
	}
	if got := merged["node-3"]; got != "http://node-3.example.internal:11434" {
		t.Fatalf("expected cli node-3 to be present, got %q", got)
	}
}

func TestLoadRegistryMissingFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "missing.yaml")

	reg, err := LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry returned error for missing file: %v", err)
	}
	if len(reg.Remotes) != 0 {
		t.Fatalf("expected empty registry, got %d remotes", len(reg.Remotes))
	}
}

func TestSaveCreatesParentDirectories(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "remotes.yaml")

	reg := &RemoteRegistry{}
	if err := reg.Add("gpu-host", "http://gpu-host.example.internal:11434"); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if err := reg.Save(path); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected registry file to exist: %v", err)
	}
}
