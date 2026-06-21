package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const DefaultEmbeddingModel = "embeddinggemma"

const transientEmbedRetryDelay = 500 * time.Millisecond

type Embedder struct {
	BaseURL string
	Model   string
	Client  *http.Client
}

func NewEmbedder(baseURL string, model string) Embedder {
	if model == "" {
		model = DefaultEmbeddingModel
	}
	return Embedder{
		BaseURL: baseURL,
		Model:   model,
		Client:  &http.Client{Timeout: 10 * time.Minute},
	}
}

func (e Embedder) Embed(ctx context.Context, text string) ([]float64, error) {
	embeddings, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("ollama embed: empty embedding response")
	}
	return embeddings[0], nil
}

func (e Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	baseURL := strings.TrimRight(e.BaseURL, "/")
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	if e.Model == "" {
		return nil, fmt.Errorf("ollama embedding model is required")
	}
	if len(texts) == 0 {
		return nil, nil
	}

	client := e.Client
	if client == nil {
		client = http.DefaultClient
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

	embeddings, err := postEmbedBatch(ctx, client, baseURL+"/api/embed", map[string]any{
		"model": e.Model,
		"input": texts,
	})
	if err == nil {
		if len(embeddings) != len(texts) {
			return nil, fmt.Errorf("ollama embed batch: expected %d embeddings, got %d", len(texts), len(embeddings))
		}
		return embeddings, nil
	}
	if !isTransientEOF(err) {
		return nil, fmt.Errorf("ollama embed batch: %w", err)
	}

	timer := time.NewTimer(transientEmbedRetryDelay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-timer.C:
	}
	embeddings, retryErr := postEmbedBatch(ctx, client, baseURL+"/api/embed", map[string]any{
		"model": e.Model,
		"input": texts,
	})
	if retryErr != nil {
		return nil, fmt.Errorf("ollama embed batch: %w; retry: %w", err, retryErr)
	}
	if len(embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama embed batch: expected %d embeddings, got %d", len(texts), len(embeddings))
	}
	return embeddings, nil
}

func isTransientEOF(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, io.EOF) {
		return true
	}
	// Ollama occasionally returns a 400 response whose JSON body contains "EOF".
	// Treat that as a transient EOF worth retrying.
	return strings.Contains(err.Error(), "EOF")
}

func postEmbedBatch(ctx context.Context, client *http.Client, url string, payload map[string]any) ([][]float64, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("post %s: %w", url, err)
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		message, err := io.ReadAll(io.LimitReader(res.Body, 4096))
		if err != nil {
			return nil, fmt.Errorf("post %s: %s: read error response: %w", url, res.Status, err)
		}
		return nil, fmt.Errorf("post %s: %s: %s", url, res.Status, strings.TrimSpace(string(message)))
	}

	const maxResponseBytes = 1 << 20
	body, err = io.ReadAll(io.LimitReader(res.Body, maxResponseBytes))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	var decoded struct {
		Embedding  []float64   `json:"embedding"`
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if len(decoded.Embeddings) > 0 {
		return decoded.Embeddings, nil
	}
	if len(decoded.Embedding) > 0 {
		return [][]float64{decoded.Embedding}, nil
	}
	return nil, fmt.Errorf("response did not include embedding: %s", strings.TrimSpace(string(body)))
}
