package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
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
	got, stats, err := parseSSEStream(strings.NewReader(body), func(c, _ string) {
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
	if stats != (streamStats{}) {
		t.Errorf("stats = %+v, want zero value", stats)
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
	_, _, err := parseSSEStream(strings.NewReader(body), func(c, r string) {
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

	got, _, err := parseSSEStream(strings.NewReader(body), nil)
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

	got, stats, err := streamChat(context.Background(), srv.URL, "m", []chatMessage{{Role: "user", Content: "yo"}}, false, nil)
	if err != nil {
		t.Fatalf("streamChat: %v", err)
	}
	if got != "hi there" {
		t.Errorf("got %q, want %q", got, "hi there")
	}
	if stats != (streamStats{}) {
		t.Errorf("stats = %+v, want zero value", stats)
	}
}

func TestStreamChatPropagatesHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not loaded", http.StatusBadRequest)
	}))
	defer srv.Close()

	_, _, err := streamChat(context.Background(), srv.URL, "m", nil, false, nil)
	if err == nil {
		t.Fatal("expected error from 400 response")
	}
	if !strings.Contains(err.Error(), "model not loaded") {
		t.Errorf("expected upstream error to surface, got %v", err)
	}
}

func TestStreamChatDefaultsThinkingOff(t *testing.T) {
	var seen chatRequest
	var decodeErr error
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decodeErr = json.NewDecoder(r.Body).Decode(&seen)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	_, _, err := streamChat(context.Background(), srv.URL, "m", []chatMessage{{Role: "user", Content: "yo"}}, false, nil)
	if err != nil {
		t.Fatalf("streamChat: %v", err)
	}
	if decodeErr != nil {
		t.Fatalf("decode request: %v", decodeErr)
	}
	if seen.ChatTemplateKwargs == nil {
		t.Fatal("expected chat_template_kwargs to be sent by default")
	}
	if got, ok := seen.ChatTemplateKwargs["enable_thinking"]; !ok || got != false {
		t.Fatalf("enable_thinking = %#v, want false", seen.ChatTemplateKwargs["enable_thinking"])
	}
}

func TestStreamChatThinkOptInOmitsThinkingKwargs(t *testing.T) {
	var seen chatRequest
	var decodeErr error
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decodeErr = json.NewDecoder(r.Body).Decode(&seen)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	_, _, err := streamChat(context.Background(), srv.URL, "m", []chatMessage{{Role: "user", Content: "yo"}}, true, nil)
	if err != nil {
		t.Fatalf("streamChat: %v", err)
	}
	if decodeErr != nil {
		t.Fatalf("decode request: %v", decodeErr)
	}
	if seen.ChatTemplateKwargs != nil {
		t.Fatalf("expected no chat_template_kwargs when --think is set, got %#v", seen.ChatTemplateKwargs)
	}
}

func TestParseSSEStreamCapturesUsageAndTimings(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"hello"}}]}`,
		``,
		`data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],"usage":{"prompt_tokens":12,"completion_tokens":247,"total_tokens":259},"timings":{"prompt_n":12,"predicted_n":247,"predicted_ms":5849.5,"predicted_per_second":42.3}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	got, stats, err := parseSSEStream(strings.NewReader(body), nil)
	if err != nil {
		t.Fatalf("parseSSEStream: %v", err)
	}
	if got != "hello" {
		t.Fatalf("assembled = %q, want hello", got)
	}
	if stats.PromptTokens != 12 || stats.CompletionTokens != 247 {
		t.Fatalf("stats tokens = %+v, want 12 prompt and 247 completion", stats)
	}
	if stats.TokensPerSecond != 42.3 {
		t.Fatalf("stats tok/s = %.2f, want 42.3", stats.TokensPerSecond)
	}
	if summary := formatTokenSummary(stats); summary != "[12 prompt + 247 generated tokens, 42.3 tok/s]" {
		t.Fatalf("summary = %q", summary)
	}
}

func TestWriteDimmedLineUsesANSIWhenSupported(t *testing.T) {
	var buf bytes.Buffer
	writeDimmedLine(&buf, true, "[1 prompt + 2 generated tokens, 3.4 tok/s]")
	want := ansiDim + "[1 prompt + 2 generated tokens, 3.4 tok/s]" + ansiReset + "\n"
	if buf.String() != want {
		t.Fatalf("got %q, want %q", buf.String(), want)
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

	err := chatLoop("http://unused", "test-model", false, interrupt, in, &out)
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
			`{"choices":[{"finish_reason":"stop","index":0,"delta":{}}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5},"timings":{"prompt_n":4,"predicted_n":1,"predicted_ms":25,"predicted_per_second":40}}`,
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

	err := chatLoop(srv.URL, "test-model", false, interrupt, in, &out)
	if err != nil {
		t.Fatalf("chatLoop: %v", err)
	}
	if !strings.Contains(out.String(), "pong") {
		t.Errorf("expected pong in output, got %q", out.String())
	}
	if !strings.Contains(out.String(), "[4 prompt + 1 generated tokens, 40.0 tok/s]") {
		t.Errorf("expected token summary in output, got %q", out.String())
	}
}

func TestIsExitCommand(t *testing.T) {
	cases := map[string]bool{
		"/bye":   true,
		"/exit":  true,
		"/quit":  true,
		"exit":   true,
		"quit":   true,
		"EXIT":   true,
		"  /bye": true,
		"/Bye":   true,
		"hello":  false,
		"":       false,
		"/by":    false,
	}
	for in, want := range cases {
		if got := isExitCommand(in); got != want {
			t.Errorf("isExitCommand(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestIsConnectionError(t *testing.T) {
	cases := map[string]bool{
		`dial tcp 127.0.0.1:11434: connect: connection refused`: true,
		`read tcp 127.0.0.1: connection reset by peer`:          true,
		`write tcp 127.0.0.1: broken pipe`:                      true,
		`unexpected EOF`:                                        true,
		`HTTP 500: model not loaded`:                            false,
		``:                                                      false,
	}
	for in, want := range cases {
		err := error(nil)
		if in != "" {
			err = fmt.Errorf("%s", in)
		}
		if got := isConnectionError(err); got != want {
			t.Errorf("isConnectionError(%q) = %v, want %v", in, got, want)
		}
	}
	if isConnectionError(nil) {
		t.Error("isConnectionError(nil) should be false")
	}
}

func TestChatLoopExitsOnConnectionRefused(t *testing.T) {
	// Pick a port the kernel will refuse so the very first POST fails.
	endpoint := "http://127.0.0.1:1" // privileged port, nothing listening
	in := strings.NewReader("hello\n")
	var out bytes.Buffer
	interrupt := make(chan struct{})

	err := chatLoop(endpoint, "test-model", false, interrupt, in, &out)
	if err != nil {
		t.Fatalf("chatLoop returned %v, want nil", err)
	}
	if !strings.Contains(out.String(), "Server stopped. Exiting.") {
		t.Errorf("expected graceful exit message, got %q", out.String())
	}
}

func TestChatLoopExitsOnInterruptAtPrompt(t *testing.T) {
	// Block-forever stdin so the only way out is the interrupt.
	in, _ := io.Pipe()
	defer in.Close()
	var out bytes.Buffer
	interrupt := make(chan struct{}, 1)

	done := make(chan error, 1)
	go func() { done <- chatLoop("http://unused", "m", false, interrupt, in, &out) }()

	// Give chatLoop a tick to reach the prompt before signaling.
	time.Sleep(50 * time.Millisecond)
	interrupt <- struct{}{}

	select {
	case err := <-done:
		if err != nil {
			t.Errorf("chatLoop returned %v, want nil", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("chatLoop did not exit on interrupt at prompt")
	}
}

func TestChatLoopExitsOnExitAlias(t *testing.T) {
	for _, alias := range []string{"/bye", "/exit", "/quit", "exit", "quit"} {
		t.Run(alias, func(t *testing.T) {
			in := strings.NewReader(alias + "\n")
			var out bytes.Buffer
			interrupt := make(chan struct{})
			if err := chatLoop("http://unused", "m", false, interrupt, in, &out); err != nil {
				t.Fatalf("chatLoop returned %v", err)
			}
		})
	}
}
