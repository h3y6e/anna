// Package openai embeds text via any OpenAI-compatible /v1/embeddings
// endpoint: llama.cpp (llama-server), Ollama, vLLM, LM Studio, or the
// OpenAI API itself.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/h3y6e/anna/internal/core"
)

const transientEmbedRetryDelay = 500 * time.Millisecond

type Embedder struct {
	url            string
	model          string
	apiKey         string
	queryPrefix    string
	documentPrefix string
}

func New(baseURL string, model string, apiKey string, queryPrefix string, documentPrefix string) (Embedder, error) {
	baseURL = strings.TrimRight(baseURL, "/")
	if baseURL == "" {
		return Embedder{}, fmt.Errorf("embedding base URL is required")
	}
	if model == "" {
		return Embedder{}, fmt.Errorf("embedding model is required")
	}
	return Embedder{
		url:            baseURL + "/v1/embeddings",
		model:          model,
		apiKey:         apiKey,
		queryPrefix:    queryPrefix,
		documentPrefix: documentPrefix,
	}, nil
}

func (e Embedder) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	embeddings, err := e.embed(ctx, []string{e.queryPrefix + text})
	if err != nil {
		return nil, err
	}
	return embeddings[0], nil
}

func (e Embedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	prefixed := make([]string, len(texts))
	for i, text := range texts {
		prefixed[i] = e.documentPrefix + text
	}
	return e.embed(ctx, prefixed)
}

func (e Embedder) embed(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		timeout := min(
			2*time.Minute+time.Duration(len(texts))*10*time.Second,
			10*time.Minute,
		)
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	embeddings, err := e.postEmbeddings(ctx, texts)
	if err == nil {
		return checkCount(embeddings, len(texts))
	}
	if !isTransientEOF(err) {
		return nil, fmt.Errorf("embed batch: %w", err)
	}

	timer := time.NewTimer(transientEmbedRetryDelay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-timer.C:
	}
	embeddings, retryErr := e.postEmbeddings(ctx, texts)
	if retryErr != nil {
		return nil, fmt.Errorf("embed batch: %w; retry: %w", err, retryErr)
	}
	return checkCount(embeddings, len(texts))
}

func checkCount(embeddings [][]float64, want int) ([][]float64, error) {
	if len(embeddings) != want {
		return nil, fmt.Errorf("embed batch: expected %d embeddings, got %d", want, len(embeddings))
	}
	return embeddings, nil
}

// isTransientEOF matches Ollama's occasional 400 response whose JSON body
// contains "EOF"; one retry is enough in practice.
func isTransientEOF(err error) bool {
	if err == nil {
		return false
	}
	return errors.Is(err, io.EOF) || strings.Contains(err.Error(), "EOF")
}

func (e Embedder) postEmbeddings(ctx context.Context, texts []string) ([][]float64, error) {
	body, err := json.Marshal(map[string]any{
		"model": e.model,
		"input": texts,
	})
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("post %s: %w", e.url, err)
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		message, err := io.ReadAll(io.LimitReader(res.Body, 4096))
		if err != nil {
			return nil, fmt.Errorf("post %s: %s: read error response: %w", e.url, res.Status, err)
		}
		if isContextOverflowError(message) {
			return nil, fmt.Errorf("post %s: %s: %s: %w", e.url, res.Status, strings.TrimSpace(string(message)), core.ErrEmbedTextTooLarge)
		}
		return nil, fmt.Errorf("post %s: %s: %s", e.url, res.Status, strings.TrimSpace(string(message)))
	}

	const maxResponseBytes = 1 << 20
	body, err = io.ReadAll(io.LimitReader(res.Body, maxResponseBytes))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	var decoded struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if len(decoded.Data) == 0 {
		return nil, fmt.Errorf("response did not include embedding: %s", strings.TrimSpace(string(body)))
	}
	sort.Slice(decoded.Data, func(i, j int) bool {
		return decoded.Data[i].Index < decoded.Data[j].Index
	})
	embeddings := make([][]float64, len(decoded.Data))
	for i, item := range decoded.Data {
		embeddings[i] = item.Embedding
	}
	return embeddings, nil
}

// isContextOverflowError matches llama.cpp error shapes for oversized input;
// the indexer splits the document and retries on core.ErrEmbedTextTooLarge.
func isContextOverflowError(body []byte) bool {
	var decoded struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return false
	}
	switch decoded.Error.Type {
	case "exceed_context_size_error":
		return true
	case "server_error":
		return strings.Contains(decoded.Error.Message, "increase the physical batch size")
	default:
		return false
	}
}
