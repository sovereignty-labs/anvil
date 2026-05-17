// Package model provides GGUF file parsing and model inventory.
package model

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

// GGUF magic number: "GGUF" in little-endian.
var ggufMagic = [4]byte{'G', 'G', 'U', 'F'}

// GGUF value types.
const (
	GGUFTypeUint8   uint32 = 0
	GGUFTypeInt8    uint32 = 1
	GGUFTypeUint16  uint32 = 2
	GGUFTypeInt16   uint32 = 3
	GGUFTypeUint32  uint32 = 4
	GGUFTypeInt32   uint32 = 5
	GGUFTypeFloat32 uint32 = 6
	GGUFTypeBool    uint32 = 7
	GGUFTypeString  uint32 = 8
	GGUFTypeArray   uint32 = 9
	GGUFTypeUint64  uint32 = 10
	GGUFTypeInt64   uint32 = 11
	GGUFTypeFloat64 uint32 = 12
)

// GGUFMetadata holds parsed metadata from a GGUF file header.
type GGUFMetadata struct {
	// File info
	Version       uint32
	TensorCount   uint64
	MetadataKV    uint64
	FileSizeBytes int64

	// All metadata key-value pairs (string keys -> typed values).
	// Values are Go-native types: string, uint8, int8, uint16, int16,
	// uint32, int32, uint64, int64, float32, float64, bool, or []any for arrays.
	KV map[string]any

	// --- Extracted fields (convenience, derived from KV) ---

	// general.*
	Architecture  string // e.g. "gemma2", "qwen2moe", "llama"
	Name          string // e.g. "Gemma 4 26B A4B"
	ContextLength uint64 // from general.context_length or arch-specific field
	FileType      uint32 // GGUF file type enum (encodes quantization)

	// Quantization (derived from FileType)
	QuantName string // e.g. "Q4_K_S", "Q3_K_XL", "BF16"

	// Architecture details
	ParamCount      uint64 // total parameter count if available
	EmbeddingLength uint64
	BlockCount      uint64
	HeadCount       uint64
	HeadCountKV     uint64

	// MoE fields
	ExpertCount     uint64 // 0 if dense model
	ExpertUsedCount uint64 // active experts per token
	IsMoE           bool

	// Tokenizer
	HasChatTemplate bool   // true if tokenizer.chat_template exists
	ChatTemplate    string // the raw Jinja template (can be large)
}

// fileTypeNames maps GGUF file type enum values to human-readable names.
// Source: ggml-common.h in llama.cpp
var fileTypeNames = map[uint32]string{
	0:  "F32",
	1:  "F16",
	2:  "Q4_0",
	3:  "Q4_1",
	6:  "Q5_0",
	7:  "Q5_1",
	8:  "Q8_0",
	9:  "Q8_1",
	10: "Q2_K",
	11: "Q3_K_S",
	12: "Q3_K_M",
	13: "Q3_K_L",
	14: "Q4_K_S",
	15: "Q4_K_M",
	16: "Q5_K_S",
	17: "Q5_K_M",
	18: "Q6_K",
	19: "IQ2_XXS",
	20: "IQ2_XS",
	21: "IQ3_XXS",
	22: "IQ1_S",
	23: "IQ4_NL",
	24: "IQ3_S",
	25: "IQ2_S",
	26: "IQ4_XS",
	27: "IQ1_M",
	28: "BF16",
	29: "Q4_0_4_4",
	30: "Q4_0_4_8",
	31: "Q4_0_8_8",
	32: "TQ1_0",
	33: "TQ2_0",
	// Non-standard / extended types that appear in community quants
	// Q3_K_XL etc. are not in the enum -- they use Q3_K_L as base type
	// and the actual quant identity comes from the filename or tensor inspection.
}

var quantPattern = regexp.MustCompile(`(?i)[_-]((?:IQ|Q)\d+(?:_K)?(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4)(?:[_.-]|$)`)

// ParseGGUF reads the GGUF header and metadata from the given file path.
// It does NOT load tensor data -- only the metadata key-value pairs.
func ParseGGUF(path string) (*GGUFMetadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open GGUF: %w", err)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat GGUF: %w", err)
	}

	m := &GGUFMetadata{
		FileSizeBytes: stat.Size(),
		KV:            make(map[string]any),
	}

	// Buffer reads — GGUF headers have hundreds of small KV reads.
	// 128KB buffer collapses thousands of syscalls into a few reads.
	br := bufio.NewReaderSize(f, 128*1024)

	r := &ggufReader{r: br, order: binary.LittleEndian}

	// Magic
	var magic [4]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("not a GGUF file (magic: %x)", magic)
	}

	// Version
	m.Version = r.readU32()
	if r.err != nil {
		return nil, fmt.Errorf("read version: %w", r.err)
	}
	if m.Version < 2 || m.Version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d (expected 2 or 3)", m.Version)
	}

	// Tensor count and metadata KV count
	if m.Version == 2 {
		m.TensorCount = uint64(r.readU32())
		m.MetadataKV = uint64(r.readU32())
	} else {
		m.TensorCount = r.readU64()
		m.MetadataKV = r.readU64()
	}
	if r.err != nil {
		return nil, fmt.Errorf("read header counts: %w", r.err)
	}

	if m.MetadataKV > 100_000 {
		return nil, fmt.Errorf("implausible metadata count: %d", m.MetadataKV)
	}

	// Parse metadata KV pairs
	for i := uint64(0); i < m.MetadataKV; i++ {
		key := r.readString()
		if r.err != nil {
			return nil, fmt.Errorf("read key %d: %w", i, r.err)
		}
		valType := r.readU32()
		if r.err != nil {
			return nil, fmt.Errorf("read value type for %q: %w", key, r.err)
		}
		val := r.readValue(valType)
		if r.err != nil {
			return nil, fmt.Errorf("read value for %q (type %d): %w", key, valType, r.err)
		}
		m.KV[key] = val
	}

	m.ExtractFields()
	return m, nil
}

func (m *GGUFMetadata) ExtractFields() {
	m.Architecture = m.getString("general.architecture")
	m.Name = m.getString("general.name")

	if v, ok := m.getUint("general.file_type"); ok {
		m.FileType = uint32(v)
		if name, exists := fileTypeNames[m.FileType]; exists {
			m.QuantName = name
		} else {
			m.QuantName = fmt.Sprintf("unknown(%d)", m.FileType)
		}
	}

	arch := m.Architecture
	if v, ok := m.getUint("general.context_length"); ok {
		m.ContextLength = v
	} else if v, ok := m.getUint(arch + ".context_length"); ok {
		m.ContextLength = v
	}

	m.EmbeddingLength, _ = m.getUint(arch + ".embedding_length")
	m.BlockCount, _ = m.getUint(arch + ".block_count")
	m.HeadCount, _ = m.getUint(arch + ".attention.head_count")
	m.HeadCountKV, _ = m.getUint(arch + ".attention.head_count_kv")

	if v, ok := m.getUint(arch + ".expert_count"); ok && v > 0 {
		m.ExpertCount = v
		m.ExpertUsedCount, _ = m.getUint(arch + ".expert_used_count")
		m.IsMoE = true
	}

	m.ParamCount, _ = m.getUint("general.parameter_count")

	if tmpl := m.getString("tokenizer.chat_template"); tmpl != "" {
		m.HasChatTemplate = true
		m.ChatTemplate = tmpl
	}
}

func (m *GGUFMetadata) getString(key string) string {
	if v, ok := m.KV[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func (m *GGUFMetadata) getUint(key string) (uint64, bool) {
	v, ok := m.KV[key]
	if !ok {
		return 0, false
	}
	switch val := v.(type) {
	case uint8:
		return uint64(val), true
	case int8:
		return uint64(val), true
	case uint16:
		return uint64(val), true
	case int16:
		return uint64(val), true
	case uint32:
		return uint64(val), true
	case int32:
		return uint64(val), true
	case uint64:
		return val, true
	case int64:
		return uint64(val), true
	case float32:
		return uint64(val), true
	case float64:
		return uint64(val), true
	case bool:
		if val {
			return 1, true
		}
		return 0, true
	default:
		return 0, false
	}
}

func (m *GGUFMetadata) EstimateParamCount() uint64 {
	if m.ParamCount > 0 {
		return m.ParamCount
	}
	return 0
}

func (m *GGUFMetadata) FileSizeGB() float64 {
	return float64(m.FileSizeBytes) / (1024 * 1024 * 1024)
}

func (m *GGUFMetadata) ArchDisplayName() string {
	arch := m.Architecture
	if arch == "" {
		arch = "unknown"
	}
	if len(arch) > 0 {
		arch = strings.ToUpper(arch[:1]) + arch[1:]
	}
	if m.IsMoE {
		return fmt.Sprintf("%s (MoE, %d experts, %d active)",
			arch, m.ExpertCount, m.ExpertUsedCount)
	}
	return arch
}

func (m *GGUFMetadata) QuantDisplayName(filename string) string {
	if filename != "" {
		if matches := quantPattern.FindStringSubmatch(strings.ToUpper(filename)); len(matches) == 2 {
			return matches[1]
		}
	}
	if m.QuantName != "" {
		return m.QuantName
	}
	return "unknown"
}

// --- GGUF binary reader ---

type ggufReader struct {
	r     io.Reader
	order binary.ByteOrder
	err   error
}

func (g *ggufReader) readU8() uint8 {
	if g.err != nil {
		return 0
	}
	var v uint8
	g.err = binary.Read(g.r, g.order, &v)
	return v
}

func (g *ggufReader) readI8() int8 {
	if g.err != nil {
		return 0
	}
	var v int8
	g.err = binary.Read(g.r, g.order, &v)
	return v
}

func (g *ggufReader) readU16() uint16 {
	if g.err != nil {
		return 0
	}
	var v uint16
	g.err = binary.Read(g.r, g.order, &v)
	return v
}

func (g *ggufReader) readI16() int16 {
	if g.err != nil {
		return 0
	}
	var v int16
	g.err = binary.Read(g.r, g.order, &v)
	return v
}

func (g *ggufReader) readU32() uint32 {
	if g.err != nil {
		return 0
	}
	var v uint32
	g.err = binary.Read(g.r, g.order, &v)
	return v
}

func (g *ggufReader) readI32() int32 {
	if g.err != nil {
		return 0
	}
	var v int32
	g.err = binary.Read(g.r, g.order, &v)
	return v
}

func (g *ggufReader) readU64() uint64 {
	if g.err != nil {
		return 0
	}
	var v uint64
	g.err = binary.Read(g.r, g.order, &v)
	return v
}

func (g *ggufReader) readI64() int64 {
	if g.err != nil {
		return 0
	}
	var v int64
	g.err = binary.Read(g.r, g.order, &v)
	return v
}

func (g *ggufReader) readF32() float32 {
	if g.err != nil {
		return 0
	}
	var v float32
	g.err = binary.Read(g.r, g.order, &v)
	return v
}

func (g *ggufReader) readF64() float64 {
	if g.err != nil {
		return 0
	}
	var v float64
	g.err = binary.Read(g.r, g.order, &v)
	return v
}

func (g *ggufReader) readBool() bool {
	return g.readU8() != 0
}

func (g *ggufReader) readString() string {
	if g.err != nil {
		return ""
	}
	length := g.readU64()
	if g.err != nil {
		return ""
	}
	if length > 10_000_000 {
		g.err = fmt.Errorf("string too long: %d bytes", length)
		return ""
	}
	buf := make([]byte, length)
	_, g.err = io.ReadFull(g.r, buf)
	if g.err != nil {
		return ""
	}
	return string(buf)
}

func (g *ggufReader) readValue(vtype uint32) any {
	if g.err != nil {
		return nil
	}
	switch vtype {
	case GGUFTypeUint8:
		return g.readU8()
	case GGUFTypeInt8:
		return g.readI8()
	case GGUFTypeUint16:
		return g.readU16()
	case GGUFTypeInt16:
		return g.readI16()
	case GGUFTypeUint32:
		return g.readU32()
	case GGUFTypeInt32:
		return g.readI32()
	case GGUFTypeFloat32:
		return g.readF32()
	case GGUFTypeBool:
		return g.readBool()
	case GGUFTypeString:
		return g.readString()
	case GGUFTypeUint64:
		return g.readU64()
	case GGUFTypeInt64:
		return g.readI64()
	case GGUFTypeFloat64:
		return g.readF64()
	case GGUFTypeArray:
		return g.readArray()
	default:
		g.err = fmt.Errorf("unknown GGUF value type: %d", vtype)
		return nil
	}
}

func (g *ggufReader) readArray() []any {
	if g.err != nil {
		return nil
	}
	elemType := g.readU32()
	count := g.readU64()
	if g.err != nil {
		return nil
	}
	if count > 10_000_000 {
		g.err = fmt.Errorf("array too large: %d elements", count)
		return nil
	}
	arr := make([]any, count)
	for i := uint64(0); i < count; i++ {
		arr[i] = g.readValue(elemType)
		if g.err != nil {
			return nil
		}
	}
	return arr
}
