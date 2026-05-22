package main

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseSSEStreamAssemblesContent(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Hello"}}]}`,
		``,
		`data: {"choices":[{"delta":{"content":" world"}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	var tokens []string
	got, err := parseSSEStream(strings.NewReader(body), func(c, _ string) {
		if c != "" {
			tokens = append(tokens, c)
		}
	})
	if err != nil {
		t.Fatalf("parseSSEStream: %v", err)
	}
	if got != "Hello world" {
		t.Errorf("assembled = %q, want %q", got, "Hello world")
	}
	if strings.Join(tokens, "|") != "Hello| world" {
		t.Errorf("tokens = %v", tokens)
	}
}

func TestParseSSEStreamHandlesReasoning(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}`,
		`data: {"choices":[{"delta":{"content":"answer"}}]}`,
		`data: [DONE]`,
	}, "\n")

	var reasoned []string
	var content []string
	_, err := parseSSEStream(strings.NewReader(body), func(c, r string) {
		if r != "" {
			reasoned = append(reasoned, r)
		}
		if c != "" {
			content = append(content, c)
		}
	})
	if err != nil {
		t.Fatalf("parseSSEStream: %v", err)
	}
	if strings.Join(reasoned, "") != "thinking..." {
		t.Errorf("reasoning = %v", reasoned)
	}
	if strings.Join(content, "") != "answer" {
		t.Errorf("content = %v", content)
	}
}

func TestParseSSEStreamSkipsMalformedChunks(t *testing.T) {
	body := strings.Join([]string{
		`data: {garbage`,
		`data: {"choices":[{"delta":{"content":"ok"}}]}`,
		`data: [DONE]`,
	}, "\n")

	got, err := parseSSEStream(strings.NewReader(body), nil)
	if err != nil {
		t.Fatalf("parseSSEStream: %v", err)
	}
	if got != "ok" {
		t.Errorf("assembled = %q, want ok", got)
	}
}

func TestStreamChatEndToEnd(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "wrong path", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		for _, chunk := range []string{
			`{"choices":[{"delta":{"content":"hi"}}]}`,
			`{"choices":[{"delta":{"content":" there"}}]}`,
			`[DONE]`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	got, err := streamChat(context.Background(), srv.URL, "m", []chatMessage{{Role: "user", Content: "yo"}}, nil)
	if err != nil {
		t.Fatalf("streamChat: %v", err)
	}
	if got != "hi there" {
		t.Errorf("got %q, want %q", got, "hi there")
	}
}

func TestStreamChatPropagatesHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not loaded", http.StatusBadRequest)
	}))
	defer srv.Close()

	_, err := streamChat(context.Background(), srv.URL, "m", nil, nil)
	if err == nil {
		t.Fatal("expected error from 400 response")
	}
	if !strings.Contains(err.Error(), "model not loaded") {
		t.Errorf("expected upstream error to surface, got %v", err)
	}
}

func TestRenderStateANSITransitions(t *testing.T) {
	var buf bytes.Buffer
	r := &renderState{supportsANSI: true}

	r.write(&buf, "", "thinking")
	r.write(&buf, "answer", "")
	r.closeReasoning(&buf)

	want := ansiDim + "thinking" + ansiReset + "answer"
	if buf.String() != want {
		t.Errorf("got %q, want %q", buf.String(), want)
	}
}

func TestRenderStateNoANSIUsesPlainPrefix(t *testing.T) {
	var buf bytes.Buffer
	r := &renderState{supportsANSI: false}
	r.write(&buf, "", "think")
	r.write(&buf, "ans", "")
	got := buf.String()
	if !strings.Contains(got, "[thinking] think") {
		t.Errorf("expected [thinking] prefix, got %q", got)
	}
	if !strings.Contains(got, "ans") {
		t.Errorf("expected answer content, got %q", got)
	}
}

func TestReadUserInputBackslashContinuation(t *testing.T) {
	in := bufio.NewReader(strings.NewReader("first \\\nsecond\n"))
	got, err := readUserInput(in)
	if err != nil {
		t.Fatalf("readUserInput: %v", err)
	}
	want := "first \nsecond"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestReadUserInputEOF(t *testing.T) {
	in := bufio.NewReader(strings.NewReader(""))
	_, err := readUserInput(in)
	if err != io.EOF {
		t.Errorf("expected EOF on empty input, got %v", err)
	}
}

func TestChatLoopByeCommand(t *testing.T) {
	in := strings.NewReader("/bye\n")
	var out bytes.Buffer
	interrupt := make(chan struct{})

	err := chatLoop("http://unused", "test-model", interrupt, in, &out)
	if err != nil {
		t.Fatalf("chatLoop: %v", err)
	}
	if !strings.Contains(out.String(), "Chatting with test-model") {
		t.Errorf("missing welcome line: %q", out.String())
	}
}

func TestChatLoopReceivesReplyFromFakeServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		for _, chunk := range []string{
			`{"choices":[{"delta":{"content":"pong"}}]}`,
			`[DONE]`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	in := strings.NewReader("ping\n/bye\n")
	var out bytes.Buffer
	interrupt := make(chan struct{})

	err := chatLoop(srv.URL, "test-model", interrupt, in, &out)
	if err != nil {
		t.Fatalf("chatLoop: %v", err)
	}
	if !strings.Contains(out.String(), "pong") {
		t.Errorf("expected pong in output, got %q", out.String())
	}
}
