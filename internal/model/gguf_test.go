package model

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func writeTestGGUF(t *testing.T, dir string, name string, kvs []testKV) string {
	t.Helper()
	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w := binary.LittleEndian
	f.Write([]byte("GGUF"))
	binary.Write(f, w, uint32(3))
	binary.Write(f, w, uint64(0))
	binary.Write(f, w, uint64(len(kvs)))
	for _, kv := range kvs {
		binary.Write(f, w, uint64(len(kv.key)))
		f.Write([]byte(kv.key))
		binary.Write(f, w, kv.vtype)
		switch kv.vtype {
		case GGUFTypeString:
			s := kv.value.(string)
			binary.Write(f, w, uint64(len(s)))
			f.Write([]byte(s))
		case GGUFTypeUint32:
			binary.Write(f, w, kv.value.(uint32))
		case GGUFTypeUint64:
			binary.Write(f, w, kv.value.(uint64))
		case GGUFTypeBool:
			if kv.value.(bool) {
				binary.Write(f, w, uint8(1))
			} else {
				binary.Write(f, w, uint8(0))
			}
		case GGUFTypeFloat32:
			binary.Write(f, w, kv.value.(float32))
		default:
			t.Fatalf("unsupported test KV type: %d", kv.vtype)
		}
	}
	return path
}

type testKV struct {
	key   string
	vtype uint32
	value any
}

func TestParseGGUF_DenseModel(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir, "test-llama-7b-Q4_K_M.gguf", []testKV{
		{"general.architecture", GGUFTypeString, "llama"},
		{"general.name", GGUFTypeString, "LLaMA 7B"},
		{"general.context_length", GGUFTypeUint64, uint64(4096)},
		{"general.file_type", GGUFTypeUint32, uint32(15)},
		{"llama.embedding_length", GGUFTypeUint32, uint32(4096)},
		{"llama.block_count", GGUFTypeUint32, uint32(32)},
		{"llama.attention.head_count", GGUFTypeUint32, uint32(32)},
		{"llama.attention.head_count_kv", GGUFTypeUint32, uint32(32)},
		{"tokenizer.chat_template", GGUFTypeString, "{% for message in messages %}...{% endfor %}"},
	})
	meta, err := ParseGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.Architecture != "llama" {
		t.Errorf("arch = %q, want llama", meta.Architecture)
	}
	if meta.Name != "LLaMA 7B" {
		t.Errorf("name = %q", meta.Name)
	}
	if meta.ContextLength != 4096 {
		t.Errorf("context = %d", meta.ContextLength)
	}
	if meta.QuantName != "Q4_K_M" {
		t.Errorf("quant = %q", meta.QuantName)
	}
	if !meta.HasChatTemplate {
		t.Error("expected HasChatTemplate = true")
	}
	if meta.IsMoE {
		t.Error("expected IsMoE = false")
	}
	if meta.EmbeddingLength != 4096 {
		t.Errorf("embedding = %d", meta.EmbeddingLength)
	}
	if meta.BlockCount != 32 {
		t.Errorf("blocks = %d", meta.BlockCount)
	}
	if meta.HeadCount != 32 {
		t.Errorf("heads = %d", meta.HeadCount)
	}
	if meta.Version != 3 {
		t.Errorf("version = %d", meta.Version)
	}
}

func TestParseGGUF_MoEModel(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir, "gemma-4-26B-A4B-Q3_K_XL.gguf", []testKV{
		{"general.architecture", GGUFTypeString, "gemma4"},
		{"general.name", GGUFTypeString, "Gemma 4 26B A4B"},
		{"general.context_length", GGUFTypeUint64, uint64(131072)},
		{"general.file_type", GGUFTypeUint32, uint32(13)},
		{"gemma4.embedding_length", GGUFTypeUint32, uint32(3072)},
		{"gemma4.block_count", GGUFTypeUint32, uint32(44)},
		{"gemma4.attention.head_count", GGUFTypeUint32, uint32(32)},
		{"gemma4.attention.head_count_kv", GGUFTypeUint32, uint32(8)},
		{"gemma4.expert_count", GGUFTypeUint32, uint32(64)},
		{"gemma4.expert_used_count", GGUFTypeUint32, uint32(2)},
		{"tokenizer.chat_template", GGUFTypeString, "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}{% endfor %}"},
	})
	meta, err := ParseGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.Architecture != "gemma4" {
		t.Errorf("arch = %q", meta.Architecture)
	}
	if !meta.IsMoE {
		t.Error("expected IsMoE = true")
	}
	if meta.ExpertCount != 64 {
		t.Errorf("expert_count = %d", meta.ExpertCount)
	}
	if meta.ExpertUsedCount != 2 {
		t.Errorf("expert_used_count = %d", meta.ExpertUsedCount)
	}
	if meta.ContextLength != 131072 {
		t.Errorf("context = %d", meta.ContextLength)
	}
	if !meta.HasChatTemplate {
		t.Error("expected HasChatTemplate = true")
	}
	qname := meta.QuantDisplayName("gemma-4-26B-A4B-Q3_K_XL.gguf")
	if qname != "Q3_K_XL" {
		t.Errorf("QuantDisplayName = %q", qname)
	}
	display := meta.ArchDisplayName()
	if display != "Gemma4 (MoE, 64 experts, 2 active)" {
		t.Errorf("ArchDisplayName = %q", display)
	}
}

func TestParseGGUF_NoTemplate(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir, "base-model.gguf", []testKV{
		{"general.architecture", GGUFTypeString, "llama"},
		{"general.name", GGUFTypeString, "Base Model"},
		{"general.file_type", GGUFTypeUint32, uint32(2)},
	})
	meta, err := ParseGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.HasChatTemplate {
		t.Error("expected HasChatTemplate = false")
	}
	if meta.QuantName != "Q4_0" {
		t.Errorf("quant = %q", meta.QuantName)
	}
}

func TestParseGGUF_BF16(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir, "model-BF16.gguf", []testKV{
		{"general.architecture", GGUFTypeString, "llama"},
		{"general.file_type", GGUFTypeUint32, uint32(28)},
	})
	meta, err := ParseGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.QuantName != "BF16" {
		t.Errorf("quant = %q", meta.QuantName)
	}
}

func TestParseGGUF_NotGGUF(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "notgguf.bin")
	os.WriteFile(path, []byte("this is not a GGUF file"), 0644)
	_, err := ParseGGUF(path)
	if err == nil {
		t.Error("expected error for non-GGUF file")
	}
}

func TestParseGGUF_ArchContextFallback(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir, "qwen-model.gguf", []testKV{
		{"general.architecture", GGUFTypeString, "qwen2moe"},
		{"qwen2moe.context_length", GGUFTypeUint32, uint32(32768)},
		{"general.file_type", GGUFTypeUint32, uint32(14)},
	})
	meta, err := ParseGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.ContextLength != 32768 {
		t.Errorf("context = %d, want 32768", meta.ContextLength)
	}
}

func TestQuantDisplayNameFromFilenameRegex(t *testing.T) {
	meta := &GGUFMetadata{QuantName: "Q5_1"}

	cases := []struct {
		filename string
		want     string
	}{
		{"model-IQ3_S.gguf", "IQ3_S"},
		{"model-Q8_K_XL.gguf", "Q8_K_XL"},
		{"model-MXFP4.gguf", "MXFP4"},
		{"model-Q4_0_4_8.gguf", "Q4_0_4_8"},
		{"model-Q3_K_XL.gguf", "Q3_K_XL"},
		{"model-BF16.gguf", "BF16"},
	}

	for _, tc := range cases {
		if got := meta.QuantDisplayName(tc.filename); got != tc.want {
			t.Errorf("QuantDisplayName(%q) = %q, want %q", tc.filename, got, tc.want)
		}
	}
}

func TestQuantDisplayNameFallsBackToEnum(t *testing.T) {
	meta := &GGUFMetadata{QuantName: "Q5_1"}
	if got := meta.QuantDisplayName("model-without-quant.gguf"); got != "Q5_1" {
		t.Fatalf("QuantDisplayName fallback = %q, want Q5_1", got)
	}
}
