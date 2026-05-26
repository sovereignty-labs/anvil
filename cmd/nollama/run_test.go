package main

import "testing"

func TestRunModelCommandDefinesRuntimeFlag(t *testing.T) {
	if runModelCmd.Flags().Lookup("runtime") == nil {
		t.Fatal("expected run command to define --runtime flag")
	}
}

func TestRunModelCommandDefinesGPUFlag(t *testing.T) {
	flag := runModelCmd.Flags().Lookup("gpu")
	if flag == nil {
		t.Fatal("expected run command to define --gpu flag")
	}
	if got, want := flag.DefValue, "-1"; got != want {
		t.Fatalf("gpu flag default = %q, want %q", got, want)
	}
}

func TestRunModelCommandDefinesNoThinkFlag(t *testing.T) {
	flag := runModelCmd.Flags().Lookup("no-think")
	if flag == nil {
		t.Fatal("expected run command to define --no-think flag")
	}
	if got, want := flag.DefValue, "false"; got != want {
		t.Fatalf("no-think flag default = %q, want %q", got, want)
	}
}

func TestCollectRunPassthroughAddsNoThinkOverride(t *testing.T) {
	cmd := runModelCmd
	if err := cmd.Flags().Set("no-think", "true"); err != nil {
		t.Fatalf("set no-think: %v", err)
	}
	defer func() {
		_ = cmd.Flags().Set("no-think", "false")
	}()

	got := collectRunPassthrough(cmd, []string{"model.gguf"})
	want := []string{"--chat-template-kwargs", `{"enable_thinking": false}`}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}
