// Package llamacpp embeds text via a llama.cpp server (llama-server) using
// its OpenAI-compatible /v1/embeddings endpoint. In router mode the server
// loads the model named in each request.
package llamacpp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

const (
	DefaultBaseURL        = "http://localhost:8080"
	DefaultEmbeddingModel = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
)

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
		return nil, fmt.Errorf("llama.cpp embed: empty embedding response")
	}
	return embeddings[0], nil
}

func (e Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	baseURL := strings.TrimRight(e.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	if e.Model == "" {
		return nil, fmt.Errorf("llama.cpp embedding model is required")
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

	embeddings, err := postEmbeddings(ctx, client, baseURL+"/v1/embeddings", map[string]any{
		"model": e.Model,
		"input": texts,
	})
	if err != nil {
		return nil, fmt.Errorf("llama.cpp embed batch: %w", err)
	}
	if len(embeddings) != len(texts) {
		return nil, fmt.Errorf("llama.cpp embed batch: expected %d embeddings, got %d", len(texts), len(embeddings))
	}
	return embeddings, nil
}

func postEmbeddings(ctx context.Context, client *http.Client, url string, payload map[string]any) ([][]float64, error) {
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
