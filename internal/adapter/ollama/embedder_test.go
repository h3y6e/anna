package ollama

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestNewEmbedderUsesEmbeddingGemmaByDefault(t *testing.T) {
	t.Parallel()

	embedder := NewEmbedder("http://ollama.example", "")
	if embedder.Model != "embeddinggemma" {
		t.Fatalf("embedding model = %q, want embeddinggemma", embedder.Model)
	}
	if embedder.BaseURL != "http://ollama.example" {
		t.Fatalf("base URL = %q, want http://ollama.example", embedder.BaseURL)
	}
}

func TestNewEmbedderUsesConfiguredModel(t *testing.T) {
	t.Parallel()

	embedder := NewEmbedder("http://ollama.example", "qwen3-embedding")
	if embedder.Model != "qwen3-embedding" {
		t.Fatalf("embedding model = %q, want qwen3-embedding", embedder.Model)
	}
}

func TestEmbedUsesCurrentEmbedEndpoint(t *testing.T) {
	t.Parallel()

	requestErr := make(chan error, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			requestErr <- fmt.Errorf("request path = %q, want /api/embed", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}

		var payload struct {
			Model string          `json:"model"`
			Input json.RawMessage `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			requestErr <- fmt.Errorf("decode request: %w", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if payload.Model != "embeddinggemma" {
			requestErr <- fmt.Errorf("model = %q, want embeddinggemma", payload.Model)
			http.Error(w, "unexpected model", http.StatusBadRequest)
			return
		}
		var inputs []string
		if err := json.Unmarshal(payload.Input, &inputs); err != nil || len(inputs) != 1 || inputs[0] != "hello" {
			requestErr <- fmt.Errorf("input = %q, want [hello]", payload.Input)
			http.Error(w, "unexpected input", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string][][]float64{
			"embeddings": {{1, 0}},
		}); err != nil {
			requestErr <- fmt.Errorf("encode response: %w", err)
			return
		}
		requestErr <- nil
	}))
	t.Cleanup(server.Close)

	embedding, err := NewEmbedder(server.URL, "").Embed(t.Context(), "hello")
	if err != nil {
		t.Fatalf("Embed error = %v", err)
	}
	if err := <-requestErr; err != nil {
		t.Fatal(err)
	}
	if len(embedding) != 2 || embedding[0] != 1 || embedding[1] != 0 {
		t.Fatalf("embedding = %#v, want [1 0]", embedding)
	}
}

func TestEmbedBatchUsesArrayInput(t *testing.T) {
	t.Parallel()

	requestErr := make(chan error, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			requestErr <- fmt.Errorf("request path = %q, want /api/embed", r.URL.Path)
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}

		var payload struct {
			Model string          `json:"model"`
			Input json.RawMessage `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			requestErr <- fmt.Errorf("decode request: %w", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		var inputs []string
		if err := json.Unmarshal(payload.Input, &inputs); err != nil || len(inputs) != 2 {
			requestErr <- fmt.Errorf("input = %q, want 2 entries", payload.Input)
			http.Error(w, "unexpected input", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string][][]float64{
			"embeddings": {{1, 0}, {0, 1}},
		}); err != nil {
			requestErr <- fmt.Errorf("encode response: %w", err)
			return
		}
		requestErr <- nil
	}))
	t.Cleanup(server.Close)

	embeddings, err := NewEmbedder(server.URL, "").EmbedBatch(t.Context(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("EmbedBatch error = %v", err)
	}
	if err := <-requestErr; err != nil {
		t.Fatal(err)
	}
	if len(embeddings) != 2 {
		t.Fatalf("embeddings count = %d, want 2", len(embeddings))
	}
}

func TestEmbedRetriesTransientEOF(t *testing.T) {
	t.Parallel()

	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			http.Error(w, `{"error":"do embedding request: EOF"}`, http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string][][]float64{
			"embeddings": {{1, 0}},
		}); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	embedding, err := NewEmbedder(server.URL, "").Embed(t.Context(), "hello")
	if err != nil {
		t.Fatalf("Embed error = %v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
	if len(embedding) != 2 || embedding[0] != 1 || embedding[1] != 0 {
		t.Fatalf("embedding = %#v, want [1 0]", embedding)
	}
}
