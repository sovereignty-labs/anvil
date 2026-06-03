package main

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sovereignty-labs/anvil/internal/model"
	"github.com/spf13/cobra"
)

type quantGGUFKV struct {
	key   string
	vtype uint32
	value any
}

func writeQuantTestGGUF(t *testing.T, dir, name string, kvs []quantGGUFKV) string {
	t.Helper()

	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	w := binary.LittleEndian
	if _, err := f.Write([]byte("GGUF")); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, w, uint32(3)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, w, uint64(0)); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, w, uint64(len(kvs))); err != nil {
		t.Fatal(err)
	}
	for _, kv := range kvs {
		if err := binary.Write(f, w, uint64(len(kv.key))); err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write([]byte(kv.key)); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(f, w, kv.vtype); err != nil {
			t.Fatal(err)
		}
		switch kv.vtype {
		case model.GGUFTypeString:
			s := kv.value.(string)
			if err := binary.Write(f, w, uint64(len(s))); err != nil {
				t.Fatal(err)
			}
			if _, err := f.Write([]byte(s)); err != nil {
				t.Fatal(err)
			}
		case model.GGUFTypeUint32:
			if err := binary.Write(f, w, kv.value.(uint32)); err != nil {
				t.Fatal(err)
			}
		case model.GGUFTypeUint64:
			if err := binary.Write(f, w, kv.value.(uint64)); err != nil {
				t.Fatal(err)
			}
		default:
			t.Fatalf("unsupported test KV type: %d", kv.vtype)
		}
	}

	return path
}

func captureStdout(t *testing.T, fn func() error) (string, error) {
	t.Helper()

	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe stdout: %v", err)
	}
	os.Stdout = w
	defer func() {
		os.Stdout = oldStdout
	}()

	outCh := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		outCh <- buf.String()
	}()

	runErr := fn()
	_ = w.Close()

	out := <-outCh
	_ = r.Close()
	return out, runErr
}

func writeCleanConfigHome(t *testing.T) {
	t.Helper()

	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", home)

	cfgDir := filepath.Join(home, "anvil")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(cfgDir, "config.yaml"), []byte("# test config\n"), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
}

func lineWithPrefix(output, prefix string) string {
	for _, line := range strings.Split(output, "\n") {
		if strings.HasPrefix(line, prefix) {
			return line
		}
	}
	return ""
}

func TestRunLoadUsesFilenameQuantDisplayName(t *testing.T) {
	writeCleanConfigHome(t)

	serverPath := filepath.Join(t.TempDir(), "llama-server")
	if err := os.WriteFile(serverPath, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatalf("write llama-server: %v", err)
	}
	t.Setenv("ANVIL_LLAMA_SERVER", serverPath)

	modelPath := writeQuantTestGGUF(t, t.TempDir(), "regression-IQ4_NL.gguf", []quantGGUFKV{
		{key: "general.architecture", vtype: model.GGUFTypeString, value: "llama"},
		{key: "general.name", vtype: model.GGUFTypeString, value: "Quant Regression"},
		{key: "general.file_type", vtype: model.GGUFTypeUint32, value: uint32(25)},
	})

	cmd := newLoadTestCommand(t)
	if err := cmd.Flags().Set("dry-run", "true"); err != nil {
		t.Fatalf("set dry-run: %v", err)
	}

	out, err := captureStdout(t, func() error {
		return runLoad(cmd, []string{modelPath})
	})
	if err != nil {
		t.Fatalf("runLoad failed: %v\noutput:\n%s", err, out)
	}

	modelLine := lineWithPrefix(out, "  Model:")
	if modelLine == "" {
		t.Fatalf("missing model line in output:\n%s", out)
	}
	if !strings.Contains(modelLine, "IQ4_NL") {
		t.Fatalf("model line did not use filename quant: %q", modelLine)
	}
	if strings.Contains(modelLine, "IQ2_S") {
		t.Fatalf("model line still used enum quant: %q", modelLine)
	}
}

func TestRunInspectUsesFilenameQuantDisplayName(t *testing.T) {
	writeCleanConfigHome(t)

	modelPath := writeQuantTestGGUF(t, t.TempDir(), "lfm2moe-8x7b-Q8_0.gguf", []quantGGUFKV{
		{key: "general.architecture", vtype: model.GGUFTypeString, value: "llama"},
		{key: "general.name", vtype: model.GGUFTypeString, value: "Inspect Regression"},
		{key: "general.file_type", vtype: model.GGUFTypeUint32, value: uint32(7)},
	})

	out, err := captureStdout(t, func() error {
		return runInspect(&cobra.Command{Use: "inspect"}, []string{modelPath})
	})
	if err != nil {
		t.Fatalf("runInspect failed: %v\noutput:\n%s", err, out)
	}

	quantLine := lineWithPrefix(out, "  Quant:")
	if quantLine == "" {
		t.Fatalf("missing quant line in output:\n%s", out)
	}
	if !strings.Contains(quantLine, "Q8_0") {
		t.Fatalf("quant line did not use filename quant: %q", quantLine)
	}
	if strings.Contains(quantLine, "Q5_1") {
		t.Fatalf("quant line still used enum quant: %q", quantLine)
	}
}
