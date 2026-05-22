package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// chatMessage is one entry in the OpenAI-style messages array.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatRequest is the JSON body POSTed to /v1/chat/completions.
type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

// streamDelta is the slice of an SSE chunk we care about. Many fields are
// ignored — we only need the deltas to print them as they arrive.
type streamDelta struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
	} `json:"choices"`
}

// streamChat POSTs messages to endpoint/v1/chat/completions with stream=true,
// parses the SSE response chunk by chunk, and invokes onToken with each
// piece of (content, reasoning) as it arrives. Returns the assembled
// assistant response text on success.
//
// The ctx is honored mid-stream: cancelling it closes the response body so a
// long generation can be interrupted (used for Ctrl+C during a reply).
func streamChat(ctx context.Context, endpoint, modelName string, messages []chatMessage, onToken func(content, reasoning string)) (string, error) {
	body, err := json.Marshal(chatRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   true,
	})
	if err != nil {
		return "", fmt.Errorf("encode chat request: %w", err)
	}

	url := strings.TrimRight(endpoint, "/") + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("post chat: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4*1024))
		return "", fmt.Errorf("chat endpoint returned %s: %s", resp.Status, strings.TrimSpace(string(errBody)))
	}

	return parseSSEStream(resp.Body, onToken)
}

// parseSSEStream reads an SSE stream from r, invoking onToken with each
// content / reasoning delta. Returns the full assembled content string.
//
// SSE format expected:
//
//	data: {"choices":[{"delta":{"content":"hi"}}]}\n
//	\n
//	data: [DONE]\n
//
// Blank lines between events are tolerated. Non-`data:` lines are ignored.
func parseSSEStream(r io.Reader, onToken func(content, reasoning string)) (string, error) {
	scanner := bufio.NewScanner(r)
	// SSE chunks can be larger than the default 64 KiB scanner buffer when
	// reasoning tokens accumulate. Give it room to breathe.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var assembled strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			break
		}
		var delta streamDelta
		if err := json.Unmarshal([]byte(payload), &delta); err != nil {
			// Tolerate junk so a single malformed chunk doesn't kill the stream.
			continue
		}
		for _, choice := range delta.Choices {
			c, rc := choice.Delta.Content, choice.Delta.ReasoningContent
			if c == "" && rc == "" {
				continue
			}
			if c != "" {
				assembled.WriteString(c)
			}
			if onToken != nil {
				onToken(c, rc)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		// Context cancellation surfaces here as a "context canceled" error;
		// treat that as a clean stop with whatever we already buffered.
		if errors.Is(err, context.Canceled) {
			return assembled.String(), nil
		}
		return assembled.String(), err
	}
	return assembled.String(), nil
}

// renderToken writes content / reasoning to w with ANSI dim for reasoning
// when supportsANSI is true. Returns true if the previous chunk was reasoning,
// so the caller can manage whitespace boundaries across calls.
type renderState struct {
	inReasoning  bool
	supportsANSI bool
}

const (
	ansiDim   = "\033[2m"
	ansiReset = "\033[0m"
)

func (r *renderState) write(w io.Writer, content, reasoning string) {
	if reasoning != "" {
		if !r.inReasoning {
			if r.supportsANSI {
				fmt.Fprint(w, ansiDim)
			} else {
				fmt.Fprint(w, "[thinking] ")
			}
			r.inReasoning = true
		}
		fmt.Fprint(w, reasoning)
	}
	if content != "" {
		if r.inReasoning {
			if r.supportsANSI {
				fmt.Fprint(w, ansiReset)
			} else {
				fmt.Fprintln(w)
			}
			r.inReasoning = false
		}
		fmt.Fprint(w, content)
	}
}

// closeReasoning emits the trailing ANSI reset (or newline) so the next prompt
// isn't dimmed when a reply ends mid-reasoning.
func (r *renderState) closeReasoning(w io.Writer) {
	if !r.inReasoning {
		return
	}
	if r.supportsANSI {
		fmt.Fprint(w, ansiReset)
	} else {
		fmt.Fprintln(w)
	}
	r.inReasoning = false
}

// chatLoop runs an interactive chat session against endpoint until the user
// types /bye, sends EOF (Ctrl+D), or cancels via interrupt.
//
// interrupt is a channel that the caller wires up to SIGINT: each receive on
// the channel cancels the in-flight stream (if any) without exiting the loop.
// A second interrupt while there's no in-flight stream — i.e. at the prompt —
// returns nil to exit cleanly.
func chatLoop(endpoint, modelName string, interrupt <-chan struct{}, in io.Reader, out io.Writer) error {
	supportsANSI := isTTY(out)
	fmt.Fprintf(out, "Chatting with %s. Type /bye to exit.\n", modelName)

	messages := make([]chatMessage, 0, 32)
	reader := bufio.NewReader(in)
	for {
		fmt.Fprint(out, ">>> ")
		line, err := readUserInput(reader)
		if err != nil {
			if errors.Is(err, io.EOF) {
				fmt.Fprintln(out)
				return nil
			}
			return err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "/bye" {
			return nil
		}
		messages = append(messages, chatMessage{Role: "user", Content: line})

		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		go func() {
			select {
			case <-interrupt:
				cancel()
			case <-done:
			}
		}()

		state := renderState{supportsANSI: supportsANSI}
		reply, err := streamChat(ctx, endpoint, modelName, messages, func(content, reasoning string) {
			state.write(out, content, reasoning)
		})
		state.closeReasoning(out)
		fmt.Fprintln(out)
		close(done)
		cancel()

		if err != nil {
			fmt.Fprintf(out, "[error] %s\n", err)
			// Roll back the user turn so the next attempt isn't poisoned by an
			// orphaned message with no assistant reply.
			messages = messages[:len(messages)-1]
			continue
		}
		messages = append(messages, chatMessage{Role: "assistant", Content: reply})
	}
}

// readUserInput reads a single line from r, supporting multi-line input via
// trailing backslash continuation.
func readUserInput(r *bufio.Reader) (string, error) {
	var b strings.Builder
	for {
		line, err := r.ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return "", err
		}
		eof := errors.Is(err, io.EOF)
		line = strings.TrimRight(line, "\n")
		if strings.HasSuffix(line, `\`) {
			b.WriteString(strings.TrimSuffix(line, `\`))
			b.WriteByte('\n')
			if eof {
				return b.String(), nil
			}
			continue
		}
		b.WriteString(line)
		if eof && b.Len() == 0 {
			return "", io.EOF
		}
		return b.String(), nil
	}
}

// isTTY reports whether w is a terminal (and therefore worth coloring).
func isTTY(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	info, err := f.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}
