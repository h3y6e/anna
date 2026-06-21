package fs

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/h3y6e/anna/internal/core"
	bolt "go.etcd.io/bbolt"
)

func TestIndexStoreSaveAndLoad(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	index := &core.Index{
		Version:        core.IndexVersion,
		Source:         "notes",
		EmbeddingModel: "model-a",
		Documents: []core.Document{
			{
				Path: "note.md",

				Content:     "a local note",
				ContentHash: "hash-a",
				Terms:       map[string]int{"note": 1},
				Length:      1,
				Embedding:   []float64{1, 0},
			},
		},
	}

	if err := store.Save(t.Context(), path, index); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	loaded, err := store.Load(t.Context(), path)
	if err != nil {
		t.Fatalf("Load error = %v", err)
	}
	if got := loaded.Documents[0].Path; got != "note.md" {
		t.Fatalf("loaded document path = %q, want note.md", got)
	}
	if got := loaded.EmbeddingModel; got != "model-a" {
		t.Fatalf("loaded embedding model = %q, want model-a", got)
	}
	if got := loaded.Documents[0].ContentHash; got != "hash-a" {
		t.Fatalf("loaded content hash = %q, want hash-a", got)
	}
	if got := loaded.Documents[0].Terms["note"]; got != 1 {
		t.Fatalf("loaded term frequency = %d, want 1", got)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat error = %v", err)
	}
	if got := info.Mode().Perm(); got != 0o600 {
		t.Fatalf("index file mode = %v, want 0600", got)
	}

	db, err := bolt.Open(path, 0o600, &bolt.Options{ReadOnly: true})
	if err != nil {
		t.Fatalf("open saved index as bbolt: %v", err)
	}
	defer db.Close()
	if err := db.View(func(tx *bolt.Tx) error {
		meta := tx.Bucket(indexMetaBucket)
		if meta == nil {
			return os.ErrNotExist
		}
		if value := meta.Get([]byte("tokenizer")); value != nil {
			return fmt.Errorf("unexpected tokenizer metadata %q", string(value))
		}
		if tx.Bucket(indexDocumentInfoBucket) == nil {
			return os.ErrNotExist
		}
		if tx.Bucket(indexManifestBucket) == nil {
			return os.ErrNotExist
		}
		if tx.Bucket(indexPostingsBucket) == nil {
			return os.ErrNotExist
		}
		if tx.Bucket(indexEmbeddingsBucket) == nil {
			return os.ErrNotExist
		}
		return nil
	}); err != nil {
		t.Fatalf("saved index buckets missing: %v", err)
	}
}

func TestIndexStoreLoadManifest(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	index := &core.Index{
		Version:        core.IndexVersion,
		Source:         "notes",
		EmbeddingModel: "model-a",
		Documents: []core.Document{
			{
				Path: "note.md",

				Content:     "a local note",
				ContentHash: "hash-a",
				Terms:       map[string]int{"note": 1},
				Length:      1,
				Embedding:   []float64{1, 0},
			},
		},
	}

	if err := store.Save(t.Context(), path, index); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	manifest, err := store.LoadManifest(t.Context(), path)
	if err != nil {
		t.Fatalf("LoadManifest error = %v", err)
	}
	if got := manifest.DocumentCount; got != 1 {
		t.Fatalf("manifest document count = %d, want 1", got)
	}
	if got := manifest.Documents["note.md"].ContentHash; got != "hash-a" {
		t.Fatalf("manifest content hash = %q, want hash-a", got)
	}
	if got := manifest.EmbeddingModel; got != "model-a" {
		t.Fatalf("manifest embedding model = %q, want model-a", got)
	}
}

func TestIndexStoreSearchUsesOptimizedBuckets(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	index := &core.Index{
		Version: core.IndexVersion,
		Source:  "notes",
		Documents: []core.Document{
			{
				Path: "alpha.md",

				Content:   "alpha note",
				Terms:     map[string]int{"alpha": 2},
				Length:    2,
				Embedding: []float64{1, 0},
			},
			{
				Path: "beta.md",

				Content:   "beta note",
				Terms:     map[string]int{"beta": 1},
				Length:    1,
				Embedding: []float64{0, 1},
			},
		},
	}

	if err := store.Save(t.Context(), path, index); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	results, err := store.Search(
		t.Context(),
		path,
		"alpha",
		1,
		fixedEmbedder{embedding: []float64{1, 0}},
		fixedTokenizer{tokens: []string{"alpha"}},
		"",
		core.SearchModeHybrid,
	)
	if err != nil {
		t.Fatalf("Search error = %v", err)
	}
	if len(results) != 1 || results[0].Path != "alpha.md" {
		t.Fatalf("Search results = %+v, want alpha.md", results)
	}
}

func TestIndexStoreSearchBM25ModeDoesNotRequireEmbeddings(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	index := &core.Index{
		Version: core.IndexVersion,
		Source:  "notes",
		Documents: []core.Document{
			{
				Path: "alpha.md",

				Content: "alpha note",
				Terms:   map[string]int{"alpha": 1},
				Length:  1,
			},
		},
	}

	if err := store.Save(t.Context(), path, index); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	results, err := store.Search(
		t.Context(),
		path,
		"alpha",
		1,
		nil,
		fixedTokenizer{tokens: []string{"alpha"}},
		"different-model",
		core.SearchModeBM25,
	)
	if err != nil {
		t.Fatalf("Search error = %v", err)
	}
	if len(results) != 1 || results[0].Path != "alpha.md" {
		t.Fatalf("Search results = %+v, want alpha.md", results)
	}
}

func TestIndexStoreSearchRejectsEmbeddingModelMismatchBeforeEmbedding(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	index := &core.Index{
		Version:        core.IndexVersion,
		Source:         "notes",
		EmbeddingModel: "model-a",
		Documents: []core.Document{
			{
				Path:      "note.md",
				Content:   "content",
				Terms:     map[string]int{"content": 1},
				Length:    1,
				Embedding: []float64{1, 0},
			},
		},
	}

	if err := store.Save(t.Context(), path, index); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	_, err := store.Search(
		t.Context(),
		path,
		"content",
		1,
		errorEmbedder{},
		fixedTokenizer{tokens: []string{"content"}},
		"model-b",
		core.SearchModeHybrid,
	)
	if err == nil || !strings.Contains(err.Error(), "index was built with embedding model model-a") {
		t.Fatalf("Search error = %v, want embedding model mismatch", err)
	}
}

func TestIndexStoreSaveRequiresIndex(t *testing.T) {
	t.Parallel()

	err := IndexStore{}.Save(t.Context(), filepath.Join(t.TempDir(), "memory.db"), nil)
	if err == nil || !strings.Contains(err.Error(), "index is required") {
		t.Fatalf("Save error = %v, want index is required", err)
	}
}

type fixedEmbedder struct {
	embedding []float64
}

func (e fixedEmbedder) Embed(context.Context, string) ([]float64, error) {
	return e.embedding, nil
}

func (e fixedEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i := range out {
		out[i] = e.embedding
	}
	return out, nil
}

type errorEmbedder struct{}

func (errorEmbedder) Embed(context.Context, string) ([]float64, error) {
	return nil, fmt.Errorf("embedder should not be called")
}

func (errorEmbedder) EmbedBatch(context.Context, []string) ([][]float64, error) {
	return nil, fmt.Errorf("embedder should not be called")
}

type fixedTokenizer struct {
	tokens []string
}

func (t fixedTokenizer) TokenizeDocument(context.Context, string) ([]string, error) {
	return t.tokens, nil
}

func (t fixedTokenizer) TokenizeQuery(context.Context, string) ([]string, error) {
	return t.tokens, nil
}
