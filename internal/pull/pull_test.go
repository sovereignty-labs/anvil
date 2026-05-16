package pull

import "testing"

func TestParseSpec(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    PullSpec
		wantErr bool
	}{
		{
			name:  "valid qwen",
			input: "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S",
			want: PullSpec{
				Org:   "unsloth",
				Repo:  "Qwen3.6-35B-A3B-GGUF",
				Quant: "Q4_K_S",
			},
		},
		{
			name:  "valid gemma",
			input: "bartowski/gemma-4-26B-A4B-it-GGUF:Q3_K_XL",
			want: PullSpec{
				Org:   "bartowski",
				Repo:  "gemma-4-26B-A4B-it-GGUF",
				Quant: "Q3_K_XL",
			},
		},
		{name: "missing slash", input: "just-a-name", wantErr: true},
		{name: "missing quant", input: "org/repo", wantErr: true},
		{name: "missing orgrepo", input: ":Q4_K_S", wantErr: true},
		{name: "empty quant", input: "org/repo:", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseSpec(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ParseSpec(%q) expected error, got nil", tt.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseSpec(%q) unexpected error: %v", tt.input, err)
			}
			if got != tt.want {
				t.Fatalf("ParseSpec(%q) = %+v, want %+v", tt.input, got, tt.want)
			}
		})
	}
}

func TestMatchQuant(t *testing.T) {
	files := []GGUFFile{
		{Name: "Model-Q4_K_S.gguf"},
		{Name: "Model-Q4_K_M.gguf"},
		{Name: "Qwen3.6-27B-IQ4_XS.gguf"},
		{Name: "Model-BF16.gguf"},
		{Name: "Model-Q8_0.gguf"},
	}

	t.Run("exact", func(t *testing.T) {
		got := MatchQuant(files, "Q4_K_S")
		if len(got) != 1 || got[0].Name != "Model-Q4_K_S.gguf" {
			t.Fatalf("MatchQuant exact = %+v", got)
		}
	})

	t.Run("case insensitive", func(t *testing.T) {
		got := MatchQuant(files, "q4_k_s")
		if len(got) != 1 || got[0].Name != "Model-Q4_K_S.gguf" {
			t.Fatalf("MatchQuant case-insensitive = %+v", got)
		}
	})

	t.Run("substring", func(t *testing.T) {
		got := MatchQuant(files, "IQ4_XS")
		if len(got) != 1 || got[0].Name != "Qwen3.6-27B-IQ4_XS.gguf" {
			t.Fatalf("MatchQuant substring = %+v", got)
		}
	})

	t.Run("ambiguous", func(t *testing.T) {
		got := MatchQuant([]GGUFFile{
			{Name: "Model-Q4_K_S.gguf"},
			{Name: "Model-Q4_K_M.gguf"},
		}, "Q4")
		if len(got) != 2 {
			t.Fatalf("MatchQuant ambiguous len = %d, want 2", len(got))
		}
	})

	t.Run("no match", func(t *testing.T) {
		got := MatchQuant(files, "Q6_K")
		if len(got) != 0 {
			t.Fatalf("MatchQuant no match len = %d, want 0", len(got))
		}
	})

	t.Run("bf16", func(t *testing.T) {
		got := MatchQuant(files, "BF16")
		if len(got) != 1 || got[0].Name != "Model-BF16.gguf" {
			t.Fatalf("MatchQuant BF16 = %+v", got)
		}
	})
}

func TestIsSplitGGUFFile(t *testing.T) {
	if !IsSplitGGUFFile("model-00001-of-00003.gguf") {
		t.Fatal("expected shard filename to be detected as split")
	}
	if !IsSplitGGUFFile("nested/model-00012-of-00045.gguf") {
		t.Fatal("expected nested shard filename to be detected as split")
	}
	if IsSplitGGUFFile("model-Q4_K_S.gguf") {
		t.Fatal("expected regular filename to not be treated as split")
	}
}
