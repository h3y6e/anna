package core

import (
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestSearcherRejectsIndexWithoutEmbeddings(t *testing.T) {
	t.Parallel()

	_, err := NewSearcher(stubIndexStore{index: &Index{Documents: []Document{{Path: "note.md"}}}}, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).
		SearchFiles(t.Context(), []string{"memory.db"}, "query", 10, SearchModeHybrid)
	if err == nil || !strings.Contains(err.Error(), "has no embedding") {
		t.Fatalf("SearchFiles error = %v, want missing embedding error", err)
	}
}

func TestSearcherSearchFilesJoinsEachMemoryDirectoryToItsDocumentPaths(t *testing.T) {
	t.Parallel()

	store := stubIndexStore{index: &Index{Documents: []Document{{Path: "todo.md", Content: "todo", Terms: map[string]int{"todo": 1}, Length: 1}}}}
	results, err := NewSearcher(store, nil, fixedTokenizer{}).
		SearchFiles(t.Context(), []string{filepath.Join("work", ".anna.db"), filepath.Join("home", ".anna.db")}, "todo", 10, SearchModeBM25)
	if err != nil {
		t.Fatalf("SearchFiles error = %v", err)
	}
	got := []string{}
	for _, result := range results {
		got = append(got, result.Path)
	}
	slices.Sort(got)
	if want := []string{filepath.Join("home", "todo.md"), filepath.Join("work", "todo.md")}; !slices.Equal(got, want) {
		t.Fatalf("SearchFiles paths = %v, want %v", got, want)
	}
}

func TestSearchUsesQueryEmbedding(t *testing.T) {
	t.Parallel()

	index := &Index{Documents: []Document{
		{Path: "semantic.md", Terms: map[string]int{}, Length: 1, Embedding: []float64{1, 0}},
		{Path: "other.md", Terms: map[string]int{}, Length: 1, Embedding: []float64{0, 1}},
	}}

	results, err := NewSearcher(stubIndexStore{index: index}, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).
		SearchFiles(t.Context(), []string{"memory.db"}, "needle", 10, SearchModeHybrid)
	if err != nil {
		t.Fatalf("SearchFiles error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("result count = %d, want 1: %#v", len(results), results)
	}
	if results[0].Path != "semantic.md" {
		t.Fatalf("top result path = %q, want semantic.md", results[0].Path)
	}
}

func TestSearchTokenizedSupportsBM25ModeWithoutEmbedding(t *testing.T) {
	t.Parallel()

	index := &Index{Documents: []Document{
		{Path: "lexical.md", Terms: map[string]int{"needle": 1}, Length: 1},
		{Path: "other.md", Terms: map[string]int{"other": 1}, Length: 1},
	}}

	results := searchTokenized(index.Documents, "needle", []string{"needle"}, nil, 10, SearchModeBM25)
	if len(results) != 1 || results[0].Path != "lexical.md" {
		t.Fatalf("results = %+v, want lexical.md", results)
	}
}

func TestSearchTokenizedVectorModeIgnoresLexicalMatches(t *testing.T) {
	t.Parallel()

	index := &Index{Documents: []Document{
		{Path: "lexical.md", Terms: map[string]int{"needle": 10}, Length: 10, Embedding: []float64{0, 1}},
		{Path: "semantic.md", Terms: map[string]int{}, Length: 1, Embedding: []float64{1, 0}},
	}}

	results := searchTokenized(index.Documents, "needle", []string{"needle"}, []float64{1, 0}, 10, SearchModeVector)
	if len(results) != 1 || results[0].Path != "semantic.md" {
		t.Fatalf("results = %+v, want semantic.md", results)
	}
}

func TestSearchTokenizedHybridUsesScoreFusion(t *testing.T) {
	t.Parallel()

	index := &Index{Documents: []Document{
		{
			Path: "note.md",

			Content:   "needle appears as an exact phrase",
			Terms:     map[string]int{"needle": 1},
			Length:    1,
			Embedding: []float64{1, 0},
		},
	}}

	results := searchTokenized(index.Documents, "needle", []string{"needle"}, []float64{1, 0}, 10, SearchModeHybrid)
	if len(results) != 1 {
		t.Fatalf("results = %+v, want one result", results)
	}

	bm25Score := math.Log(1 + 0.5/1.5)
	expected := 0.8 + 0.2*(bm25Score/(bm25Score+1))
	if math.Abs(results[0].Score-expected) > 1e-10 {
		t.Fatalf("hybrid score = %.12f, want %.12f", results[0].Score, expected)
	}
}

func TestSearchTokenizedRRFUsesRRFAndCosineRescore(t *testing.T) {
	t.Parallel()

	index := &Index{Documents: []Document{
		{
			Path: "keyword-and-vector.md",

			Terms:     map[string]int{"needle": 3},
			Length:    3,
			Embedding: []float64{0.8, 0.6},
		},
		{
			Path: "keyword-only.md",

			Terms:     map[string]int{"needle": 1},
			Length:    1,
			Embedding: []float64{0, 1},
		},
		{
			Path: "vector-only.md",

			Terms:     map[string]int{},
			Length:    1,
			Embedding: []float64{1, 0},
		},
	}}

	results := searchTokenized(index.Documents, "needle", []string{"needle"}, []float64{1, 0}, 10, SearchModeRRF)
	if len(results) != 3 {
		t.Fatalf("results = %+v, want three results", results)
	}
	if results[0].Path != "keyword-and-vector.md" {
		t.Fatalf("top result path = %q, want keyword-and-vector.md", results[0].Path)
	}

	expected := 0.7 + 0.3*0.8
	if math.Abs(results[0].Score-expected) > 1e-10 {
		t.Fatalf("rrf score = %.12f, want %.12f", results[0].Score, expected)
	}
}

func TestSearcherRejectsUnsupportedMode(t *testing.T) {
	t.Parallel()

	_, err := NewSearcher(stubIndexStore{}, nil, fixedTokenizer{}).
		SearchFiles(t.Context(), []string{"memory.db"}, "query", 10, SearchMode("unknown"))
	if err == nil || !strings.Contains(err.Error(), `unsupported search mode "unknown"`) {
		t.Fatalf("SearchFiles error = %v, want unsupported mode", err)
	}
}

func TestSearchCJKExactQueryDoesNotExpandToBigram(t *testing.T) {
	t.Parallel()

	index, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "tokyo.md", Content: "# 東京都\n行政区域"},
		{Path: "kyoto.md", Content: "# 京都\n旅行"},
	}}, nil, fixedEmbedder{embedding: []float64{1, 0}}, cjkTokenizer{}).build(t.Context(), "notes")
	if err != nil {
		t.Fatalf("Build error = %v", err)
	}

	results, err := NewSearcher(stubIndexStore{index: index}, nil, cjkTokenizer{}).
		SearchFiles(t.Context(), []string{"memory.db"}, "東京都", 10, SearchModeBM25)
	if err != nil {
		t.Fatalf("SearchFiles error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("result count = %d, want 1: %#v", len(results), results)
	}
	if results[0].Path != "tokyo.md" {
		t.Fatalf("top result path = %q, want tokyo.md", results[0].Path)
	}
}

func TestSearchCJKSpacedTermsMatchCompoundDocument(t *testing.T) {
	t.Parallel()

	index, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "poll.md", Content: "# 投票作成UI\n選択肢編集"},
	}}, nil, fixedEmbedder{embedding: []float64{1, 0}}, cjkTokenizer{}).build(t.Context(), "notes")
	if err != nil {
		t.Fatalf("Build error = %v", err)
	}

	results, err := NewSearcher(stubIndexStore{index: index}, nil, cjkTokenizer{}).
		SearchFiles(t.Context(), []string{"memory.db"}, "投票 作成", 10, SearchModeBM25)
	if err != nil {
		t.Fatalf("SearchFiles error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("result count = %d, want 1: %#v", len(results), results)
	}
	if results[0].Path != "poll.md" {
		t.Fatalf("top result path = %q, want poll.md", results[0].Path)
	}
}

func TestSearcherRejectsIndexBuiltWithDifferentPrefixes(t *testing.T) {
	t.Parallel()

	// Arrange
	store := stubIndexStore{index: &Index{
		Embedding: EmbeddingProfile{Model: "model"},
		Documents: []Document{{Path: "note.md", Embedding: []float64{1, 0}}},
	}}
	searcher := NewSearcher(store, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).
		WithEmbedding(EmbeddingProfile{Model: "model", QueryPrefix: "query: ", DocumentPrefix: "passage: "})

	// Act
	_, err := searcher.SearchFiles(t.Context(), []string{"memory.db"}, "query", 10, SearchModeHybrid)

	// Assert
	if err == nil || !strings.Contains(err.Error(), `--embedder-query-prefix "" --embedder-document-prefix ""`) {
		t.Fatalf("SearchFiles error = %v, want prefix mismatch", err)
	}
}
