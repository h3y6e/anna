package fs

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/h3y6e/anna/internal/core"
	bolt "go.etcd.io/bbolt"
)

func TestWhenAnIndexIsSavedLoadingItReturnsTheSameDocumentsFromAPrivateBboltFile(t *testing.T) {
	t.Parallel()

	// Arrange
	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	index := &core.Index{
		Embedding: core.EmbeddingProfile{Model: "model-a"},
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

	// Act
	if err := store.Save(t.Context(), path, index); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	loaded, err := store.Load(t.Context(), path)
	if err != nil {
		t.Fatalf("Load error = %v", err)
	}

	// Assert
	if got := loaded.Documents[0].Path; got != "note.md" {
		t.Fatalf("loaded document path = %q, want note.md", got)
	}
	if got := loaded.Embedding.Model; got != "model-a" {
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

func TestWhenAnIndexIsSavedLoadingTheManifestReturnsItsCountHashesAndModel(t *testing.T) {
	t.Parallel()

	// Arrange
	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	index := &core.Index{
		Embedding: core.EmbeddingProfile{Model: "model-a"},
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

	// Act
	if err := store.Save(t.Context(), path, index); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	manifest, err := store.LoadManifest(t.Context(), path)
	if err != nil {
		t.Fatalf("LoadManifest error = %v", err)
	}

	// Assert
	if got := manifest.DocumentCount; got != 1 {
		t.Fatalf("manifest document count = %d, want 1", got)
	}
	if got := manifest.Documents["note.md"].ContentHash; got != "hash-a" {
		t.Fatalf("manifest content hash = %q, want hash-a", got)
	}
	if got := manifest.Embedding.Model; got != "model-a" {
		t.Fatalf("manifest embedding model = %q, want model-a", got)
	}
}

func TestWhenAMemoryIsSavedSearchingItReturnsTheMatchingNoteJoinedToTheMemoryDirectory(t *testing.T) {
	t.Parallel()

	// Arrange
	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	index := &core.Index{
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

	// Act
	results, err := searchMemories(
		t.Context(),
		[]string{path},
		"alpha",
		1,
		fixedEmbedder{embedding: []float64{1, 0}},
		fixedTokenizer{tokens: []string{"alpha"}},
		core.EmbeddingProfile{},
		core.SearchModeHybrid,
	)

	// Assert
	if err != nil {
		t.Fatalf("Search error = %v", err)
	}
	if want := filepath.Join(filepath.Dir(path), "alpha.md"); len(results) != 1 || results[0].Path != want {
		t.Fatalf("Search results = %+v, want %s", results, want)
	}
}

func TestWhenSearchingInBM25ModeTheEmbedderAndEmbeddingModelAreNotNeeded(t *testing.T) {
	t.Parallel()

	// Arrange
	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	index := &core.Index{
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

	// Act
	results, err := searchMemories(
		t.Context(),
		[]string{path},
		"alpha",
		1,
		nil,
		fixedTokenizer{tokens: []string{"alpha"}},
		core.EmbeddingProfile{Model: "different-model"},
		core.SearchModeBM25,
	)

	// Assert
	if err != nil {
		t.Fatalf("Search error = %v", err)
	}
	if want := filepath.Join(filepath.Dir(path), "alpha.md"); len(results) != 1 || results[0].Path != want {
		t.Fatalf("Search results = %+v, want %s", results, want)
	}
}

func TestWhenTheConfiguredModelDiffersFromTheMemorySearchingFailsBeforeEmbeddingTheQuery(t *testing.T) {
	t.Parallel()

	// Arrange
	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	index := &core.Index{
		Embedding: core.EmbeddingProfile{Model: "model-a"},
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

	// Act
	_, err := searchMemories(
		t.Context(),
		[]string{path},
		"content",
		1,
		errorEmbedder{},
		fixedTokenizer{tokens: []string{"content"}},
		core.EmbeddingProfile{Model: "model-b"},
		core.SearchModeHybrid,
	)

	// Assert
	if err == nil || !strings.Contains(err.Error(), `index was built with --embedder-model "model-a"`) {
		t.Fatalf("Search error = %v, want embedding model mismatch", err)
	}
}

func TestWhenSeveralMemoriesHoldSameNamedNotesSearchingReturnsEachNoteUnderItsOwnDirectory(t *testing.T) {
	t.Parallel()

	// Arrange
	store := IndexStore{}
	work := filepath.Join(t.TempDir(), "work")
	home := filepath.Join(t.TempDir(), "home")
	for _, dir := range []string{work, home} {
		saveMemory(t, store, filepath.Join(dir, ".anna.db"), "", core.Document{
			Path:      "todo.md",
			Content:   "todo list",
			Terms:     map[string]int{"todo": 1},
			Length:    1,
			Embedding: []float64{1, 0},
		})
	}

	// Act
	results, err := searchMemories(
		t.Context(),
		[]string{filepath.Join(work, ".anna.db"), filepath.Join(home, ".anna.db")},
		"todo",
		10,
		fixedEmbedder{embedding: []float64{1, 0}},
		fixedTokenizer{tokens: []string{"todo"}},
		core.EmbeddingProfile{},
		core.SearchModeRRF,
	)

	// Assert
	if err != nil {
		t.Fatalf("Search error = %v", err)
	}
	got := []string{}
	for _, result := range results {
		got = append(got, result.Path)
	}
	slices.Sort(got)
	want := []string{filepath.Join(home, "todo.md"), filepath.Join(work, "todo.md")}
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Fatalf("Search paths = %v, want %v", got, want)
	}
}

func TestWhenSeveralMemoriesAreSearchedTogetherTermsAreScoredAgainstTheCombinedCorpus(t *testing.T) {
	t.Parallel()

	// Arrange
	store := IndexStore{}
	common := filepath.Join(t.TempDir(), "common")
	rare := filepath.Join(t.TempDir(), "rare")
	saveMemory(t, store, filepath.Join(common, ".anna.db"), "",
		core.Document{Path: "a.md", Content: "go", Terms: map[string]int{"go": 1}, Length: 1},
		core.Document{Path: "b.md", Content: "go", Terms: map[string]int{"go": 1}, Length: 1},
		core.Document{Path: "c.md", Content: "go", Terms: map[string]int{"go": 1}, Length: 1},
	)
	saveMemory(t, store, filepath.Join(rare, ".anna.db"), "",
		core.Document{Path: "d.md", Content: "go", Terms: map[string]int{"go": 1}, Length: 1},
		core.Document{Path: "e.md", Content: "other", Terms: map[string]int{"other": 1}, Length: 1},
		core.Document{Path: "f.md", Content: "other", Terms: map[string]int{"other": 1}, Length: 1},
	)
	search := func(paths ...string) float64 {
		results, err := searchMemories(t.Context(), paths, "go", 10, nil, fixedTokenizer{tokens: []string{"go"}}, core.EmbeddingProfile{}, core.SearchModeBM25)
		if err != nil {
			t.Fatalf("Search error = %v", err)
		}
		for _, result := range results {
			if filepath.Base(result.Path) == "d.md" {
				return result.Score
			}
		}
		t.Fatalf("Search results = %+v, want d.md", results)
		return 0
	}

	// Act
	alone := search(filepath.Join(rare, ".anna.db"))
	combined := search(filepath.Join(rare, ".anna.db"), filepath.Join(common, ".anna.db"))

	// Assert
	if combined >= alone {
		t.Fatalf("d.md score with a corpus where the term is common = %v, want below %v", combined, alone)
	}
}

func TestWhenMemoriesWereBuiltWithDifferentModelsSearchingThemTogetherFails(t *testing.T) {
	t.Parallel()

	// Arrange
	store := IndexStore{}
	a := filepath.Join(t.TempDir(), "a", ".anna.db")
	b := filepath.Join(t.TempDir(), "b", ".anna.db")
	doc := core.Document{Path: "note.md", Content: "note", Terms: map[string]int{"note": 1}, Length: 1, Embedding: []float64{1, 0}}
	saveMemory(t, store, a, "model-a", doc)
	saveMemory(t, store, b, "model-b", doc)

	// Act
	_, err := searchMemories(t.Context(), []string{a, b}, "note", 10, errorEmbedder{}, fixedTokenizer{tokens: []string{"note"}}, core.EmbeddingProfile{Model: "model-a"}, core.SearchModeHybrid)

	// Assert
	if err == nil || !strings.Contains(err.Error(), `index was built with --embedder-model "model-b"`) {
		t.Fatalf("Search error = %v, want embedding model mismatch", err)
	}
}

func searchMemories(
	ctx context.Context,
	paths []string,
	query string,
	limit int,
	embedder core.Embedder,
	tokenizer core.Tokenizer,
	embedding core.EmbeddingProfile,
	mode core.SearchMode,
) ([]core.SearchResult, error) {
	return core.NewSearcher(IndexStore{}, embedder, tokenizer).WithEmbedding(embedding).SearchFiles(ctx, paths, query, limit, mode)
}

func saveMemory(t *testing.T, store IndexStore, path string, embeddingModel string, docs ...core.Document) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatalf("mkdir memory dir: %v", err)
	}
	if err := store.Save(t.Context(), path, &core.Index{Embedding: core.EmbeddingProfile{Model: embeddingModel}, Documents: docs}); err != nil {
		t.Fatalf("save memory %s: %v", path, err)
	}
}

type fixedEmbedder struct {
	embedding []float64
}

func (e fixedEmbedder) EmbedQuery(context.Context, string) ([]float64, error) {
	return e.embedding, nil
}

func (e fixedEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i := range out {
		out[i] = e.embedding
	}
	return out, nil
}

type errorEmbedder struct{}

func (errorEmbedder) EmbedQuery(context.Context, string) ([]float64, error) {
	return nil, fmt.Errorf("embedder should not be called")
}

func (errorEmbedder) EmbedDocuments(context.Context, []string) ([][]float64, error) {
	return nil, fmt.Errorf("embedder should not be called")
}

type fixedTokenizer struct {
	tokens []string
}

func (t fixedTokenizer) Tokenize(string) []string {
	return t.tokens
}

func TestWhenAnIndexIsSavedWithPrefixesLoadingAndLoadingTheManifestReturnThem(t *testing.T) {
	t.Parallel()

	// Arrange
	path := filepath.Join(t.TempDir(), "memory.db")
	store := IndexStore{}
	profile := core.EmbeddingProfile{Model: "model-a", QueryPrefix: "query: ", DocumentPrefix: "passage: "}

	// Act
	if err := store.Save(t.Context(), path, &core.Index{Embedding: profile}); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	loaded, err := store.Load(t.Context(), path)
	if err != nil {
		t.Fatalf("Load error = %v", err)
	}
	manifest, err := store.LoadManifest(t.Context(), path)
	if err != nil {
		t.Fatalf("LoadManifest error = %v", err)
	}

	// Assert
	if loaded.Embedding != profile {
		t.Fatalf("loaded embedding profile = %+v, want %+v", loaded.Embedding, profile)
	}
	if manifest.Embedding != profile {
		t.Fatalf("manifest embedding profile = %+v, want %+v", manifest.Embedding, profile)
	}
}

func TestWhenTheConfiguredPrefixesDifferFromTheMemorySearchingFailsBeforeEmbeddingTheQuery(t *testing.T) {
	t.Parallel()

	// Arrange
	path := filepath.Join(t.TempDir(), ".anna.db")
	store := IndexStore{}
	saveMemory(t, store, path, "model-a", core.Document{Path: "note.md", Content: "note", Terms: map[string]int{"note": 1}, Length: 1, Embedding: []float64{1, 0}})
	configured := core.EmbeddingProfile{Model: "model-a", QueryPrefix: "query: "}

	// Act
	_, err := searchMemories(t.Context(), []string{path}, "note", 10, errorEmbedder{}, fixedTokenizer{tokens: []string{"note"}}, configured, core.SearchModeHybrid)

	// Assert
	if err == nil || !strings.Contains(err.Error(), `--embedder-query-prefix "" --embedder-document-prefix ""`) {
		t.Fatalf("Search error = %v, want prefix mismatch", err)
	}
}

func TestWhenTheMemoryFileIsMissingLoadingTheManifestFailsWithErrNotExist(t *testing.T) {
	t.Parallel()

	// Act
	_, err := IndexStore{}.LoadManifest(t.Context(), filepath.Join(t.TempDir(), "memory.db"))

	// Assert
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("LoadManifest error = %v, want os.ErrNotExist", err)
	}
}

func TestWhenTheMemoryHasAnotherIndexVersionLoadingTheManifestFailsWithErrIndexVersionMismatch(t *testing.T) {
	t.Parallel()

	// Arrange
	path := filepath.Join(t.TempDir(), "memory.db")
	if err := (IndexStore{}).Save(t.Context(), path, &core.Index{}); err != nil {
		t.Fatalf("Save error = %v", err)
	}
	db, err := bolt.Open(path, 0o600, nil)
	if err != nil {
		t.Fatalf("open index: %v", err)
	}
	if err := db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(indexMetaBucket).Put(indexVersionKey, []byte("4"))
	}); err != nil {
		t.Fatalf("rewrite version: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close index: %v", err)
	}

	// Act
	_, err = IndexStore{}.LoadManifest(t.Context(), path)

	// Assert
	if !errors.Is(err, core.ErrIndexVersionMismatch) {
		t.Fatalf("LoadManifest error = %v, want core.ErrIndexVersionMismatch", err)
	}
}
