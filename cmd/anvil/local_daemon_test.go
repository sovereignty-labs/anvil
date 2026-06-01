package main

import "testing"

func TestLocalDaemonURL(t *testing.T) {
	cases := map[string]string{
		"0.0.0.0:11434":   "http://127.0.0.1:11434",
		":11434":          "http://127.0.0.1:11434",
		"127.0.0.1:11434": "http://127.0.0.1:11434",
		"203.0.113.30:11434": "http://203.0.113.30:11434",
		"localhost:8000":  "http://localhost:8000",
		"":                "",
		"garbage":         "",
	}
	for in, want := range cases {
		if got := localDaemonURL(in); got != want {
			t.Errorf("localDaemonURL(%q) = %q, want %q", in, got, want)
		}
	}
}
