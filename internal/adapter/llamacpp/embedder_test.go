package llamacpp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNewEmbedderUsesEmbeddingGemmaGGUFByDefault(t *testing.T) {
	t.Parallel()

	embedder := NewEmbedder("http://llama.example", "")
	if embedder.Model != "ggml-org/embeddinggemma-300M-GGUF:Q8_0" {
		t.Fatalf("embedding model = %q, want ggml-org/embeddinggemma-300M-GGUF:Q8_0", embedder.Model)
	}
	if embedder.BaseURL != "http://llama.example" {
		t.Fatalf("base URL = %q, want http://llama.example", embedder.BaseURL)
	}
}

func TestNewEmbedderUsesConfiguredModel(t *testing.T) {
	t.Parallel()

	embedder := NewEmbedder("http://llama.example", "custom-model")
	if embedder.Model != "custom-model" {
		t.Fatalf("embedding model = %q, want custom-model", embedder.Model)
	}
}

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
		if payload.Model != "ggml-org/embeddinggemma-300M-GGUF:Q8_0" {
			requestErr <- fmt.Errorf("model = %q, want ggml-org/embeddinggemma-300M-GGUF:Q8_0", payload.Model)
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
	if embeddings[0][0] != 1 || embeddings[1][1] != 1 {
		t.Fatalf("embeddings = %#v, want [[1 0] [0 1]]", embeddings)
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

	embeddings, err := NewEmbedder(server.URL, "").EmbedBatch(t.Context(), []string{"hello", "world"})
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

	embedding, err := NewEmbedder(server.URL, "").Embed(t.Context(), "hello")
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

	_, err := NewEmbedder(server.URL, "missing").EmbedBatch(t.Context(), []string{"hello"})
	if err == nil {
		t.Fatal("EmbedBatch error = nil, want model not found error")
	}
	if want := "model 'missing' not found"; !strings.Contains(err.Error(), want) {
		t.Fatalf("EmbedBatch error = %v, want message containing %q", err, want)
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

	_, err := NewEmbedder(server.URL, "").EmbedBatch(t.Context(), []string{"hello", "world"})
	if err == nil {
		t.Fatal("EmbedBatch error = nil, want embedding count mismatch error")
	}
}
