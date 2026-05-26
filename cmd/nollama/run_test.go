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
