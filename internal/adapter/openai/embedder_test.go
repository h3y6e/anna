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

func TestWhenTheServerSpeaksTheOpenAIAPIEmbeddingDocumentsPostsToV1EmbeddingsAndReturnsTheVectors(t *testing.T) {
	t.Parallel()

	// Arrange
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

	// Act
	embeddings, err := newEmbedder(t, server.URL, "test-model", "", "", "").EmbedDocuments(t.Context(), []string{"hello", "world"})

	// Assert
	if err != nil {
		t.Fatalf("EmbedDocuments error = %v", err)
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

func TestWhenAnAPIKeyIsConfiguredEmbeddingDocumentsSendsItAsABearerToken(t *testing.T) {
	t.Parallel()

	// Arrange
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

	// Act
	if _, err := newEmbedder(t, server.URL, "test-model", "secret-key", "", "").EmbedDocuments(t.Context(), []string{"hello"}); err != nil {
		t.Fatalf("EmbedDocuments error = %v", err)
	}

	// Assert
	if got := <-authorization; got != "Bearer secret-key" {
		t.Fatalf("Authorization header = %q, want Bearer secret-key", got)
	}
}

func TestWhenNoAPIKeyIsConfiguredEmbeddingDocumentsSendsNoAuthorizationHeader(t *testing.T) {
	t.Parallel()

	// Arrange
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

	// Act
	if _, err := newEmbedder(t, server.URL, "test-model", "", "", "").EmbedDocuments(t.Context(), []string{"hello"}); err != nil {
		t.Fatalf("EmbedDocuments error = %v", err)
	}

	// Assert
	if got := <-authorization; got != "" {
		t.Fatalf("Authorization header = %q, want empty", got)
	}
}

func TestWhenTheServerReturnsEmbeddingsOutOfOrderEmbeddingDocumentsOrdersThemByIndex(t *testing.T) {
	t.Parallel()

	// Arrange
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

	// Act
	embeddings, err := newEmbedder(t, server.URL, "test-model", "", "", "").EmbedDocuments(t.Context(), []string{"hello", "world"})

	// Assert
	if err != nil {
		t.Fatalf("EmbedDocuments error = %v", err)
	}
	if embeddings[0][0] != 1 || embeddings[1][1] != 1 {
		t.Fatalf("embeddings = %#v, want ordered by index [[1 0] [0 1]]", embeddings)
	}
}

func TestWhenTheServerReturnsOneEmbeddingEmbeddingAQueryReturnsThatVector(t *testing.T) {
	t.Parallel()

	// Arrange
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

	// Act
	embedding, err := newEmbedder(t, server.URL, "test-model", "", "", "").EmbedQuery(t.Context(), "hello")

	// Assert
	if err != nil {
		t.Fatalf("EmbedQuery error = %v", err)
	}
	if len(embedding) != 2 || embedding[0] != 1 || embedding[1] != 0 {
		t.Fatalf("embedding = %#v, want [1 0]", embedding)
	}
}

func TestWhenTheServerRejectsTheRequestEmbeddingDocumentsFailsWithTheServerMessage(t *testing.T) {
	t.Parallel()

	// Arrange
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"code":400,"message":"model 'missing' not found","type":"invalid_request_error"}}`)
	}))
	t.Cleanup(server.Close)

	// Act
	_, err := newEmbedder(t, server.URL, "missing", "", "", "").EmbedDocuments(t.Context(), []string{"hello"})

	// Assert
	if err == nil {
		t.Fatal("EmbedDocuments error = nil, want model not found error")
	}
	if want := "model 'missing' not found"; !strings.Contains(err.Error(), want) {
		t.Fatalf("EmbedDocuments error = %v, want message containing %q", err, want)
	}
}

func TestWhenTheServerReportsAContextOverflowEmbeddingDocumentsFailsWithErrEmbedTextTooLarge(t *testing.T) {
	t.Parallel()

	// Arrange
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"code":400,"message":"input (2131 tokens) is larger than the max context size (2048 tokens). skipping","type":"exceed_context_size_error","n_prompt_tokens":2131,"n_ctx":2048}}`)
	}))
	t.Cleanup(server.Close)

	// Act
	_, err := newEmbedder(t, server.URL, "model", "", "", "").EmbedDocuments(t.Context(), []string{"long text"})

	// Assert
	if err == nil {
		t.Fatal("EmbedDocuments error = nil, want context overflow error")
	}
	if !errors.Is(err, core.ErrEmbedTextTooLarge) {
		t.Fatalf("EmbedDocuments error = %v, want error wrapping core.ErrEmbedTextTooLarge", err)
	}
}

func TestWhenTheBatchExceedsThePhysicalBatchSizeEmbeddingDocumentsFailsWithErrEmbedTextTooLarge(t *testing.T) {
	t.Parallel()

	// Arrange
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error":{"code":500,"message":"input (11182 tokens) is too large to process. increase the physical batch size (current batch size: 8192)","type":"server_error"}}`)
	}))
	t.Cleanup(server.Close)

	// Act
	_, err := newEmbedder(t, server.URL, "model", "", "", "").EmbedDocuments(t.Context(), []string{"a", "b"})

	// Assert
	if err == nil {
		t.Fatal("EmbedDocuments error = nil, want physical batch size error")
	}
	if !errors.Is(err, core.ErrEmbedTextTooLarge) {
		t.Fatalf("EmbedDocuments error = %v, want error wrapping core.ErrEmbedTextTooLarge", err)
	}
}

func TestWhenTheServerReturnsATransientEOFEmbeddingDocumentsRetriesOnceAndSucceeds(t *testing.T) {
	t.Parallel()

	// Arrange
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

	// Act
	embeddings, err := newEmbedder(t, server.URL, "test-model", "", "", "").EmbedDocuments(t.Context(), []string{"hello"})

	// Assert
	if err != nil {
		t.Fatalf("EmbedDocuments error = %v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("request count = %d, want 2 (one retry after transient EOF)", calls.Load())
	}
	if len(embeddings) != 1 || embeddings[0][0] != 1 {
		t.Fatalf("embeddings = %#v, want [[1 0]]", embeddings)
	}
}

func TestWhenTheServerReturnsFewerEmbeddingsThanInputsEmbeddingDocumentsFails(t *testing.T) {
	t.Parallel()

	// Arrange
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

	// Act
	_, err := newEmbedder(t, server.URL, "test-model", "", "", "").EmbedDocuments(t.Context(), []string{"hello", "world"})

	// Assert
	if err == nil {
		t.Fatal("EmbedDocuments error = nil, want embedding count mismatch error")
	}
}

func TestWhenTheModelIsEmptyCreatingAnEmbedderFailsWithModelIsRequired(t *testing.T) {
	t.Parallel()

	// Act
	_, err := New("http://embedder.example", "", "", "", "")

	// Assert
	if err == nil || !strings.Contains(err.Error(), "model is required") {
		t.Fatalf("New error = %v, want model is required", err)
	}
}

func TestWhenTheBaseURLIsEmptyCreatingAnEmbedderFailsWithBaseURLIsRequired(t *testing.T) {
	t.Parallel()

	// Act
	_, err := New("", "test-model", "", "", "")

	// Assert
	if err == nil || !strings.Contains(err.Error(), "base URL is required") {
		t.Fatalf("New error = %v, want base URL is required", err)
	}
}

func newEmbedder(t *testing.T, baseURL string, model string, apiKey string, queryPrefix string, documentPrefix string) Embedder {
	t.Helper()
	embedder, err := New(baseURL, model, apiKey, queryPrefix, documentPrefix)
	if err != nil {
		t.Fatalf("New error = %v", err)
	}
	return embedder
}

func TestWhenPrefixesAreConfiguredEmbeddingPrependsTheQueryPrefixToQueriesAndTheDocumentPrefixToDocuments(t *testing.T) {
	t.Parallel()

	// Arrange
	var inputs [][]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
		}
		inputs = append(inputs, payload.Input)
		data := make([]map[string]any, len(payload.Input))
		for i := range payload.Input {
			data[i] = map[string]any{"index": i, "embedding": []float64{1}}
		}
		if err := json.NewEncoder(w).Encode(map[string]any{"data": data}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	embedder := newEmbedder(t, server.URL, "test-model", "", "query: ", "passage: ")

	// Act
	if _, err := embedder.EmbedQuery(t.Context(), "猫"); err != nil {
		t.Fatalf("EmbedQuery error = %v", err)
	}
	if _, err := embedder.EmbedDocuments(t.Context(), []string{"a", "b"}); err != nil {
		t.Fatalf("EmbedDocuments error = %v", err)
	}

	// Assert
	want := [][]string{{"query: 猫"}, {"passage: a", "passage: b"}}
	if fmt.Sprint(inputs) != fmt.Sprint(want) {
		t.Fatalf("inputs = %q, want %q", inputs, want)
	}
}
