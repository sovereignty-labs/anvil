package main

import "testing"

func TestRunModelCommandDefinesRuntimeFlag(t *testing.T) {
	if runModelCmd.Flags().Lookup("runtime") == nil {
		t.Fatal("expected run command to define --runtime flag")
	}
}
