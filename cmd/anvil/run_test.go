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

func TestRunModelCommandDefinesThinkFlag(t *testing.T) {
	flag := runModelCmd.Flags().Lookup("think")
	if flag == nil {
		t.Fatal("expected run command to define --think flag")
	}
	if got, want := flag.DefValue, "false"; got != want {
		t.Fatalf("think flag default = %q, want %q", got, want)
	}
}
