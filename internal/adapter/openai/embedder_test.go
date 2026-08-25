package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/h3y6e/anna/internal/core"
)

func TestEmbedBatchUsesOpenAIEmbeddingsEndpoint(t *testing.T) {
	t.Parallel()

	requestErr := make(chan error, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			requestErr <- fmt.Errorf("request path = %q, want /v1/embeddings", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}

		var payload struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			requestErr <- fmt.Errorf("decode request: %w", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if payload.Model != "test-model" {
			requestErr <- fmt.Errorf("model = %q, want test-model", payload.Model)
			http.Error(w, "unexpected model", http.StatusBadRequest)
			return
		}
		if len(payload.Input) != 2 || payload.Input[0] != "hello" || payload.Input[1] != "world" {
			requestErr <- fmt.Errorf("input = %v, want [hello world]", payload.Input)
			http.Error(w, "unexpected input", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"object": "embedding", "index": 0, "embedding": []float64{1, 0}},
				{"object": "embedding", "index": 1, "embedding": []float64{0, 1}},
			},
		}); err != nil {
			requestErr <- fmt.Errorf("encode response: %w", err)
			return
		}
		requestErr <- nil
	}))
	t.Cleanup(server.Close)

	embeddings, err := NewEmbedder(server.URL, "test-model", "").EmbedBatch(t.Context(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("EmbedBatch error = %v", err)
	}
	if err := <-requestErr; err != nil {
		t.Fatal(err)
	}
	if len(embeddings) != 2 {
		t.Fatalf("embeddings count = %d, want 2", len(embeddings))
	}
	if embeddings[0][0] != 1 || embeddings[1][1] != 1 {
		t.Fatalf("embeddings = %#v, want [[1 0] [0 1]]", embeddings)
	}
}

func TestEmbedBatchSendsBearerTokenWhenAPIKeyIsConfigured(t *testing.T) {
	t.Parallel()

	authorization := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization <- r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"index": 0, "embedding": []float64{1, 0}}},
		}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	if _, err := NewEmbedder(server.URL, "test-model", "secret-key").EmbedBatch(t.Context(), []string{"hello"}); err != nil {
		t.Fatalf("EmbedBatch error = %v", err)
	}
	if got := <-authorization; got != "Bearer secret-key" {
		t.Fatalf("Authorization header = %q, want Bearer secret-key", got)
	}
}

func TestEmbedBatchOmitsAuthorizationHeaderWhenAPIKeyIsEmpty(t *testing.T) {
	t.Parallel()

	authorization := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization <- r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"index": 0, "embedding": []float64{1, 0}}},
		}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	if _, err := NewEmbedder(server.URL, "test-model", "").EmbedBatch(t.Context(), []string{"hello"}); err != nil {
		t.Fatalf("EmbedBatch error = %v", err)
	}
	if got := <-authorization; got != "" {
		t.Fatalf("Authorization header = %q, want empty", got)
	}
}

func TestEmbedBatchOrdersEmbeddingsByResponseIndex(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"index": 1, "embedding": []float64{0, 1}},
				{"index": 0, "embedding": []float64{1, 0}},
			},
		}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	embeddings, err := NewEmbedder(server.URL, "test-model", "").EmbedBatch(t.Context(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("EmbedBatch error = %v", err)
	}
	if embeddings[0][0] != 1 || embeddings[1][1] != 1 {
		t.Fatalf("embeddings = %#v, want ordered by index [[1 0] [0 1]]", embeddings)
	}
}

func TestEmbedReturnsSingleEmbedding(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"index": 0, "embedding": []float64{1, 0}},
			},
		}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	embedding, err := NewEmbedder(server.URL, "test-model", "").Embed(t.Context(), "hello")
	if err != nil {
		t.Fatalf("Embed error = %v", err)
	}
	if len(embedding) != 2 || embedding[0] != 1 || embedding[1] != 0 {
		t.Fatalf("embedding = %#v, want [1 0]", embedding)
	}
}

func TestEmbedBatchSurfacesServerErrorMessage(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"code":400,"message":"model 'missing' not found","type":"invalid_request_error"}}`)
	}))
	t.Cleanup(server.Close)

	_, err := NewEmbedder(server.URL, "missing", "").EmbedBatch(t.Context(), []string{"hello"})
	if err == nil {
		t.Fatal("EmbedBatch error = nil, want model not found error")
	}
	if want := "model 'missing' not found"; !strings.Contains(err.Error(), want) {
		t.Fatalf("EmbedBatch error = %v, want message containing %q", err, want)
	}
}

func TestEmbedBatchWrapsErrEmbedTextTooLargeWhenServerReportsContextOverflow(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"code":400,"message":"input (2131 tokens) is larger than the max context size (2048 tokens). skipping","type":"exceed_context_size_error","n_prompt_tokens":2131,"n_ctx":2048}}`)
	}))
	t.Cleanup(server.Close)

	_, err := NewEmbedder(server.URL, "model", "").EmbedBatch(t.Context(), []string{"long text"})
	if err == nil {
		t.Fatal("EmbedBatch error = nil, want context overflow error")
	}
	if !errors.Is(err, core.ErrEmbedTextTooLarge) {
		t.Fatalf("EmbedBatch error = %v, want error wrapping core.ErrEmbedTextTooLarge", err)
	}
}

func TestEmbedBatchWrapsErrEmbedTextTooLargeWhenBatchExceedsPhysicalBatchSize(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error":{"code":500,"message":"input (11182 tokens) is too large to process. increase the physical batch size (current batch size: 8192)","type":"server_error"}}`)
	}))
	t.Cleanup(server.Close)

	_, err := NewEmbedder(server.URL, "model", "").EmbedBatch(t.Context(), []string{"a", "b"})
	if err == nil {
		t.Fatal("EmbedBatch error = nil, want physical batch size error")
	}
	if !errors.Is(err, core.ErrEmbedTextTooLarge) {
		t.Fatalf("EmbedBatch error = %v, want error wrapping core.ErrEmbedTextTooLarge", err)
	}
}

func TestEmbedBatchRetriesOnceWhenServerReturnsTransientEOF(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":"EOF"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"index": 0, "embedding": []float64{1, 0}}},
		}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	embeddings, err := NewEmbedder(server.URL, "test-model", "").EmbedBatch(t.Context(), []string{"hello"})
	if err != nil {
		t.Fatalf("EmbedBatch error = %v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("request count = %d, want 2 (one retry after transient EOF)", calls.Load())
	}
	if len(embeddings) != 1 || embeddings[0][0] != 1 {
		t.Fatalf("embeddings = %#v, want [[1 0]]", embeddings)
	}
}

func TestEmbedBatchRejectsEmbeddingCountMismatch(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"index": 0, "embedding": []float64{1, 0}},
			},
		}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	_, err := NewEmbedder(server.URL, "test-model", "").EmbedBatch(t.Context(), []string{"hello", "world"})
	if err == nil {
		t.Fatal("EmbedBatch error = nil, want embedding count mismatch error")
	}
}

func TestEmbedBatchRequiresModel(t *testing.T) {
	t.Parallel()

	_, err := NewEmbedder("http://embedder.example", "", "").EmbedBatch(t.Context(), []string{"hello"})
	if err == nil || !strings.Contains(err.Error(), "model is required") {
		t.Fatalf("EmbedBatch error = %v, want model is required", err)
	}
}

func TestEmbedBatchRequiresBaseURL(t *testing.T) {
	t.Parallel()

	_, err := NewEmbedder("", "test-model", "").EmbedBatch(t.Context(), []string{"hello"})
	if err == nil || !strings.Contains(err.Error(), "base URL is required") {
		t.Fatalf("EmbedBatch error = %v, want base URL is required", err)
	}
}
