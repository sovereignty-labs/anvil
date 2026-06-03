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
	Model              string         `json:"model"`
	Messages           []chatMessage  `json:"messages"`
	Stream             bool           `json:"stream"`
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
}

// streamChunk is the slice of an SSE chunk we care about. Many fields are
// ignored — we only need the deltas to print them as they arrive, plus the
// final usage/timing metadata for the token-rate summary line.
type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
	Timings *struct {
		PromptN            int     `json:"prompt_n"`
		PredictedN         int     `json:"predicted_n"`
		PromptMS           float64 `json:"prompt_ms"`
		PredictedMS        float64 `json:"predicted_ms"`
		PromptPerSecond    float64 `json:"prompt_per_second"`
		PredictedPerSecond float64 `json:"predicted_per_second"`
	} `json:"timings,omitempty"`
}

// streamStats captures the final prompt/completion counts and generation
// speed derived from the final SSE chunk.
type streamStats struct {
	PromptTokens     int
	CompletionTokens int
	Seconds          float64
	TokensPerSecond  float64
}

// streamChat POSTs messages to endpoint/v1/chat/completions with stream=true,
// parses the SSE response chunk by chunk, and invokes onToken with each
// piece of (content, reasoning) as it arrives. Returns the assembled
// assistant response text on success.
//
// The ctx is honored mid-stream: cancelling it closes the response body so a
// long generation can be interrupted (used for Ctrl+C during a reply).
func streamChat(ctx context.Context, endpoint, modelName string, messages []chatMessage, thinkEnabled bool, onToken func(content, reasoning string)) (string, streamStats, error) {
	reqBody := chatRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   true,
	}
	if !thinkEnabled {
		reqBody.ChatTemplateKwargs = map[string]any{
			"enable_thinking": false,
		}
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", streamStats{}, fmt.Errorf("encode chat request: %w", err)
	}

	url := strings.TrimRight(endpoint, "/") + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", streamStats{}, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", streamStats{}, fmt.Errorf("post chat: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4*1024))
		return "", streamStats{}, fmt.Errorf("chat endpoint returned %s: %s", resp.Status, strings.TrimSpace(string(errBody)))
	}

	return parseSSEStream(resp.Body, onToken)
}

// parseSSEStream reads an SSE stream from r, invoking onToken with each
// content / reasoning delta. Returns the full assembled content string and
// the final prompt/completion timing metadata when present.
//
// SSE format expected:
//
//	data: {"choices":[{"delta":{"content":"hi"}}]}\n
//	\n
//	data: [DONE]\n
//
// Blank lines between events are tolerated. Non-`data:` lines are ignored.
func parseSSEStream(r io.Reader, onToken func(content, reasoning string)) (string, streamStats, error) {
	scanner := bufio.NewScanner(r)
	// SSE chunks can be larger than the default 64 KiB scanner buffer when
	// reasoning tokens accumulate. Give it room to breathe.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var assembled strings.Builder
	var finalStats streamStats
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
		var delta streamChunk
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
		if stats, ok := streamStatsFromChunk(delta); ok {
			finalStats = stats
		}
	}
	if err := scanner.Err(); err != nil {
		// Context cancellation surfaces here as a "context canceled" error;
		// treat that as a clean stop with whatever we already buffered.
		if errors.Is(err, context.Canceled) {
			return assembled.String(), finalStats, nil
		}
		return assembled.String(), finalStats, err
	}
	return assembled.String(), finalStats, nil
}

func streamStatsFromChunk(chunk streamChunk) (streamStats, bool) {
	if chunk.Usage == nil && chunk.Timings == nil {
		return streamStats{}, false
	}

	stats := streamStats{}
	if chunk.Usage != nil {
		stats.PromptTokens = chunk.Usage.PromptTokens
		stats.CompletionTokens = chunk.Usage.CompletionTokens
		if stats.PromptTokens == 0 && chunk.Usage.TotalTokens > 0 && stats.CompletionTokens > 0 {
			stats.PromptTokens = chunk.Usage.TotalTokens - stats.CompletionTokens
		}
	}
	if chunk.Timings != nil {
		if stats.PromptTokens == 0 && chunk.Timings.PromptN > 0 {
			stats.PromptTokens = chunk.Timings.PromptN
		}
		if stats.CompletionTokens == 0 && chunk.Timings.PredictedN > 0 {
			stats.CompletionTokens = chunk.Timings.PredictedN
		}
		if chunk.Timings.PredictedMS > 0 {
			stats.Seconds = chunk.Timings.PredictedMS / 1000
		}
		if chunk.Timings.PredictedPerSecond > 0 {
			stats.TokensPerSecond = chunk.Timings.PredictedPerSecond
		}
	}
	if stats.TokensPerSecond <= 0 && stats.Seconds > 0 && stats.CompletionTokens > 0 {
		stats.TokensPerSecond = float64(stats.CompletionTokens) / stats.Seconds
	}
	if stats.Seconds <= 0 && stats.TokensPerSecond > 0 && stats.CompletionTokens > 0 {
		stats.Seconds = float64(stats.CompletionTokens) / stats.TokensPerSecond
	}
	return stats, true
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

func formatTokenSummary(stats streamStats) string {
	if stats.CompletionTokens <= 0 || stats.TokensPerSecond <= 0 {
		return ""
	}
	if stats.PromptTokens < 0 {
		stats.PromptTokens = 0
	}
	return fmt.Sprintf("[%d prompt + %d generated tokens, %.1f tok/s]", stats.PromptTokens, stats.CompletionTokens, stats.TokensPerSecond)
}

func writeDimmedLine(w io.Writer, supportsANSI bool, line string) {
	if line == "" {
		return
	}
	if supportsANSI {
		fmt.Fprintf(w, "%s%s%s\n", ansiDim, line, ansiReset)
		return
	}
	fmt.Fprintln(w, line)
}

// isExitCommand recognizes the many ways users naturally try to leave the
// chat: /bye, /exit, /quit, plus bare exit/quit (case-insensitive).
func isExitCommand(input string) bool {
	switch strings.ToLower(strings.TrimSpace(input)) {
	case "/bye", "/exit", "/quit", "exit", "quit":
		return true
	}
	return false
}

// isConnectionError reports whether err looks like the backend went away.
// When llama-server dies mid-session every subsequent POST returns one of
// these — surfacing them as fatal lets the chat loop exit cleanly instead of
// spamming errors on each user input.
func isConnectionError(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "connection refused") ||
		strings.Contains(s, "connection reset") ||
		strings.Contains(s, "broken pipe") ||
		strings.Contains(s, "EOF")
}

// inputResult bundles a user-input line with its read error so the chat
// loop can select between a blocking stdin read and an interrupt.
type inputResult struct {
	line string
	err  error
}

// chatLoop runs an interactive chat session against endpoint until the user
// types an exit command, sends EOF (Ctrl+D), or sends an interrupt (Ctrl+C).
//
// interrupt is wired by the caller to SIGINT. During a streaming reply, an
// interrupt cancels the in-flight stream and the loop continues. At the
// prompt (no stream in flight), an interrupt exits cleanly. A connection
// error after streamChat means the backend died — the loop reports that and
// exits rather than spamming "connection refused" on every subsequent input.
func chatLoop(endpoint, modelName string, thinkEnabled bool, interrupt <-chan struct{}, in io.Reader, out io.Writer) error {
	supportsANSI := isTTY(out)
	fmt.Fprintf(out, "Chatting with %s. Type /bye to exit.\n", modelName)

	messages := make([]chatMessage, 0, 32)
	reader := bufio.NewReader(in)
	for {
		fmt.Fprint(out, ">>> ")

		// Read in a goroutine so the loop can also wake on interrupt while
		// the user is staring at the prompt. The bufio.Reader keeps its
		// position across iterations because it's captured by the closure.
		inputCh := make(chan inputResult, 1)
		go func() {
			line, err := readUserInput(reader)
			inputCh <- inputResult{line: line, err: err}
		}()

		var res inputResult
		select {
		case res = <-inputCh:
		case <-interrupt:
			fmt.Fprintln(out)
			return nil
		}

		if res.err != nil {
			if errors.Is(res.err, io.EOF) {
				fmt.Fprintln(out)
				return nil
			}
			return res.err
		}
		line := strings.TrimSpace(res.line)
		if line == "" {
			continue
		}
		if isExitCommand(line) {
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
		reply, stats, err := streamChat(ctx, endpoint, modelName, messages, thinkEnabled, func(content, reasoning string) {
			state.write(out, content, reasoning)
		})
		state.closeReasoning(out)
		fmt.Fprintln(out)
		writeDimmedLine(out, supportsANSI, formatTokenSummary(stats))
		close(done)
		cancel()

		if err != nil {
			// Roll back the orphaned user turn either way so the message
			// history matches what the model actually saw.
			messages = messages[:len(messages)-1]
			if isConnectionError(err) {
				fmt.Fprintln(out, "Server stopped. Exiting.")
				return nil
			}
			fmt.Fprintf(out, "[error] %s\n", err)
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
