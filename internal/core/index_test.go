package core

import (
	"context"
	"math"
	"strings"
	"testing"
)

func TestIndexerBuildRequiresTextSource(t *testing.T) {
	t.Parallel()

	_, err := NewIndexer(nil, nil, nil, nil).Build(t.Context(), "notes")
	if err == nil || !strings.Contains(err.Error(), "text source is required") {
		t.Fatalf("Build error = %v, want text source is required", err)
	}
}

func TestIndexerBuildRequiresEmbedder(t *testing.T) {
	t.Parallel()

	_, err := NewIndexer(stubTextSource{}, nil, nil, fixedTokenizer{}).Build(t.Context(), "notes")
	if err == nil || !strings.Contains(err.Error(), "embedder is required") {
		t.Fatalf("Build error = %v, want embedder is required", err)
	}
}

func TestIndexerBuildRequiresTokenizer(t *testing.T) {
	t.Parallel()

	_, err := NewIndexer(stubTextSource{}, nil, fixedEmbedder{embedding: []float64{1, 0}}, nil).Build(t.Context(), "notes")
	if err == nil || !strings.Contains(err.Error(), "tokenizer is required") {
		t.Fatalf("Build error = %v, want tokenizer is required", err)
	}
}

func TestIndexerBuildAndSaveRequiresIndexStore(t *testing.T) {
	t.Parallel()

	_, err := NewIndexer(stubTextSource{}, nil, nil, nil).BuildAndSave(t.Context(), "notes", "memory.db")
	if err == nil || !strings.Contains(err.Error(), "index store is required") {
		t.Fatalf("BuildAndSave error = %v, want index store is required", err)
	}
}

func TestIndexerBuildAndSaveReusesUnchangedDocuments(t *testing.T) {
	t.Parallel()

	unchangedContent := "# Keep\n\nsame body"
	store := &capturingIndexStore{index: &Index{
		Version:        IndexVersion,
		Source:         "notes",
		EmbeddingModel: "model",
		Documents: []Document{
			{
				Path: "keep.md",

				Content:     unchangedContent,
				ContentHash: contentHash(unchangedContent),
				Terms:       map[string]int{"keep": 1},
				Length:      1,
				Embedding:   []float64{0, 1},
			},
		},
	}}
	store.manifest = &IndexManifest{
		Version:        IndexVersion,
		Source:         "notes",
		EmbeddingModel: "model",
		DocumentCount:  1,
		Documents: map[string]DocumentManifest{
			"keep.md": {ContentHash: contentHash(unchangedContent)},
		},
	}
	embedder := &countingEmbedder{embedding: []float64{1, 0}}
	index, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "keep.md", Content: unchangedContent},
		{Path: "new.md", Content: "# New\n\nnew body"},
	}}, store, embedder, fixedTokenizer{}).WithEmbeddingModel("model").
		BuildAndSave(t.Context(), "notes", "memory.db")
	if err != nil {
		t.Fatalf("BuildAndSave error = %v", err)
	}
	if embedder.calls != 1 {
		t.Fatalf("Embed calls = %d, want 1 changed document only", embedder.calls)
	}
	if store.loadCalls != 1 {
		t.Fatalf("Load calls = %d, want 1 after manifest detects changes", store.loadCalls)
	}
	if store.saved == nil {
		t.Fatalf("Save was not called for changed index")
	}
	if got := index.Documents[0].Embedding; got[0] != 0 || got[1] != 1 {
		t.Fatalf("reused embedding = %v, want existing embedding", got)
	}
	if got := index.Documents[1].Path; got != "new.md" {
		t.Fatalf("new document path = %q, want new.md", got)
	}
}

func TestIndexerBuildAndSaveSkipsSaveWhenNothingChanged(t *testing.T) {
	t.Parallel()

	content := "# Keep\n\nsame body"
	store := &capturingIndexStore{index: &Index{
		Version:        IndexVersion,
		Source:         "notes",
		EmbeddingModel: "model",
		Documents: []Document{
			{
				Path: "keep.md",

				Content:     content,
				ContentHash: contentHash(content),
				Terms:       map[string]int{"keep": 1},
				Length:      1,
				Embedding:   []float64{0, 1},
			},
		},
	}}
	store.manifest = &IndexManifest{
		Version:        IndexVersion,
		Source:         "notes",
		EmbeddingModel: "model",
		DocumentCount:  1,
		Documents: map[string]DocumentManifest{
			"keep.md": {ContentHash: contentHash(content)},
		},
	}
	embedder := &countingEmbedder{embedding: []float64{1, 0}}
	index, err := NewIndexer(stubTextSource{files: []TextFile{{Path: "keep.md", Content: content}}}, store, embedder, fixedTokenizer{}).WithEmbeddingModel("model").
		BuildAndSave(t.Context(), "notes", "memory.db")
	if err != nil {
		t.Fatalf("BuildAndSave error = %v", err)
	}
	if embedder.calls != 0 {
		t.Fatalf("Embed calls = %d, want 0", embedder.calls)
	}
	if store.loadCalls != 0 {
		t.Fatalf("Load calls = %d, want 0", store.loadCalls)
	}
	if store.saved != nil {
		t.Fatalf("Save was called for unchanged index")
	}
	if got := index.Count(); got != 1 {
		t.Fatalf("index count = %d, want 1", got)
	}
}

func TestIndexerBuildEmbedsDocumentsInBatches(t *testing.T) {
	t.Parallel()

	embedder := &countingEmbedder{embedding: []float64{1, 0}}
	index, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "a.md", Content: "# A\n"},
		{Path: "b.md", Content: "# B\n"},
		{Path: "c.md", Content: "# C\n"},
	}}, nil, embedder, fixedTokenizer{}).WithEmbeddingModel("model").
		Build(t.Context(), "notes")
	if err != nil {
		t.Fatalf("Build error = %v", err)
	}
	if len(index.Documents) != 3 {
		t.Fatalf("document count = %d, want 3", len(index.Documents))
	}
	for _, doc := range index.Documents {
		if len(doc.Embedding) != 2 {
			t.Fatalf("embedding for %s = %#v, want len 2", doc.Path, doc.Embedding)
		}
	}
	if embedder.calls != 1 {
		t.Fatalf("embedder calls = %d, want 1 batch call", embedder.calls)
	}
}

func TestIndexerBuildAndSaveRebuildOptionIgnoresReusableIndex(t *testing.T) {
	t.Parallel()

	content := "# Keep\n\nsame body"
	store := &capturingIndexStore{index: &Index{
		Version:        IndexVersion,
		Source:         "notes",
		EmbeddingModel: "model",
		Documents: []Document{
			{
				Path:        "keep.md",
				Content:     content,
				ContentHash: contentHash(content),
				Terms:       map[string]int{"keep": 1},
				Length:      1,
				Embedding:   []float64{0, 1},
			},
		},
	}}
	embedder := &countingEmbedder{embedding: []float64{1, 0}}
	_, err := NewIndexer(stubTextSource{files: []TextFile{{Path: "keep.md", Content: content}}}, store, embedder, fixedTokenizer{}).WithEmbeddingModel("model").
		BuildAndSaveWithOptions(t.Context(), "notes", "memory.db", IndexBuildOptions{Rebuild: true})
	if err != nil {
		t.Fatalf("BuildAndSaveWithOptions error = %v", err)
	}
	if embedder.calls != 1 {
		t.Fatalf("Embed calls = %d, want full rebuild to embed document", embedder.calls)
	}
	if store.saved == nil {
		t.Fatalf("Save was not called for rebuild")
	}
}

func TestSearcherRequiresIndexStore(t *testing.T) {
	t.Parallel()

	_, err := NewSearcher(nil, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).SearchFile(t.Context(), "memory.db", "query", 10, SearchModeHybrid)
	if err == nil || !strings.Contains(err.Error(), "index store is required") {
		t.Fatalf("SearchFile error = %v, want index store is required", err)
	}
}

func TestSearcherRequiresEmbedder(t *testing.T) {
	t.Parallel()

	_, err := NewSearcher(stubIndexStore{}, nil, fixedTokenizer{}).SearchFile(t.Context(), "memory.db", "query", 10, SearchModeHybrid)
	if err == nil || !strings.Contains(err.Error(), "embedder is required") {
		t.Fatalf("SearchFile error = %v, want embedder is required", err)
	}
}

func TestSearcherRequiresTokenizer(t *testing.T) {
	t.Parallel()

	_, err := NewSearcher(stubIndexStore{}, fixedEmbedder{embedding: []float64{1, 0}}, nil).SearchFile(t.Context(), "memory.db", "query", 10, SearchModeHybrid)
	if err == nil || !strings.Contains(err.Error(), "tokenizer is required") {
		t.Fatalf("SearchFile error = %v, want tokenizer is required", err)
	}
}

func TestSearcherRejectsIndexWithoutEmbeddings(t *testing.T) {
	t.Parallel()

	_, err := NewSearcher(stubIndexStore{index: &Index{Documents: []Document{{Path: "note.md"}}}}, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).
		SearchFile(t.Context(), "memory.db", "query", 10, SearchModeHybrid)
	if err == nil || !strings.Contains(err.Error(), "has no embedding") {
		t.Fatalf("SearchFile error = %v, want missing embedding error", err)
	}
}

func TestSearchUsesQueryEmbedding(t *testing.T) {
	t.Parallel()

	index := &Index{Documents: []Document{
		{Path: "semantic.md", Terms: map[string]int{}, Length: 1, Embedding: []float64{1, 0}},
		{Path: "other.md", Terms: map[string]int{}, Length: 1, Embedding: []float64{0, 1}},
	}}

	results, err := Search(t.Context(), index, "needle", []float64{1, 0}, 10, fixedTokenizer{}, SearchModeHybrid)
	if err != nil {
		t.Fatalf("Search error = %v", err)
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

	results, err := SearchTokenized(index, "needle", []string{"needle"}, nil, 10, SearchModeBM25)
	if err != nil {
		t.Fatalf("SearchTokenized error = %v", err)
	}
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

	results, err := SearchTokenized(index, "needle", []string{"needle"}, []float64{1, 0}, 10, SearchModeVector)
	if err != nil {
		t.Fatalf("SearchTokenized error = %v", err)
	}
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

	results, err := SearchTokenized(index, "needle", []string{"needle"}, []float64{1, 0}, 10, SearchModeHybrid)
	if err != nil {
		t.Fatalf("SearchTokenized error = %v", err)
	}
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

	results, err := SearchTokenized(index, "needle", []string{"needle"}, []float64{1, 0}, 10, SearchModeRRF)
	if err != nil {
		t.Fatalf("SearchTokenized error = %v", err)
	}
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

func TestSearchTokenizedRejectsUnsupportedMode(t *testing.T) {
	t.Parallel()

	_, err := SearchTokenized(&Index{}, "query", []string{"query"}, nil, 10, SearchMode("unknown"))
	if err == nil || !strings.Contains(err.Error(), `unsupported search mode "unknown"`) {
		t.Fatalf("SearchTokenized error = %v, want unsupported mode", err)
	}
}

func TestEmbeddingInputTruncatesLongDocuments(t *testing.T) {
	t.Parallel()

	input := embeddingInput(Document{Content: strings.Repeat("あ", 8000)})
	if got := len([]rune(input)); got != 4000 {
		t.Fatalf("embedding input length = %d, want 4000", got)
	}
}

func TestSearchCJKExactQueryDoesNotExpandToBigram(t *testing.T) {
	t.Parallel()

	index, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "tokyo.md", Content: "# 東京都\n行政区域"},
		{Path: "kyoto.md", Content: "# 京都\n旅行"},
	}}, nil, fixedEmbedder{embedding: []float64{1, 0}}, cjkTokenizer{}).Build(t.Context(), "notes")
	if err != nil {
		t.Fatalf("Build error = %v", err)
	}

	results, err := Search(t.Context(), index, "東京都", nil, 10, cjkTokenizer{}, SearchModeHybrid)
	if err != nil {
		t.Fatalf("Search error = %v", err)
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
	}}, nil, fixedEmbedder{embedding: []float64{1, 0}}, cjkTokenizer{}).Build(t.Context(), "notes")
	if err != nil {
		t.Fatalf("Build error = %v", err)
	}

	results, err := Search(t.Context(), index, "投票 作成", nil, 10, cjkTokenizer{}, SearchModeHybrid)
	if err != nil {
		t.Fatalf("Search error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("result count = %d, want 1: %#v", len(results), results)
	}
	if results[0].Path != "poll.md" {
		t.Fatalf("top result path = %q, want poll.md", results[0].Path)
	}
}

func TestIndexerBuildReportsProgress(t *testing.T) {
	t.Parallel()

	var progress []IndexProgress
	_, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "a.md", Content: "# A\nbody a"},
		{Path: "b.md", Content: "# B\nbody b"},
	}}, nil, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).
		WithProgress(func(p IndexProgress) { progress = append(progress, p) }).
		Build(t.Context(), "notes")
	if err != nil {
		t.Fatalf("Build error = %v", err)
	}
	if len(progress) != 2 {
		t.Fatalf("progress calls = %d, want 2", len(progress))
	}
	if progress[0].Current != 1 || progress[0].Total != 2 || progress[0].Path != "a.md" || progress[0].Cached {
		t.Fatalf("progress[0] = %+v, want {1 2 a.md false}", progress[0])
	}
	if progress[1].Current != 2 || progress[1].Total != 2 || progress[1].Path != "b.md" || progress[1].Cached {
		t.Fatalf("progress[1] = %+v, want {2 2 b.md false}", progress[1])
	}
}

func TestIndexerIncrementalBuildReportsProgressWithCachedDocuments(t *testing.T) {
	t.Parallel()

	unchangedContent := "# Keep\n\nsame body"
	store := &capturingIndexStore{index: &Index{
		Version:        IndexVersion,
		Source:         "notes",
		EmbeddingModel: "model",
		Documents: []Document{
			{
				Path: "keep.md",

				Content:     unchangedContent,
				ContentHash: contentHash(unchangedContent),
				Terms:       map[string]int{"keep": 1},
				Length:      1,
				Embedding:   []float64{0, 1},
			},
		},
	}}
	store.manifest = &IndexManifest{
		Version:        IndexVersion,
		Source:         "notes",
		EmbeddingModel: "model",
		DocumentCount:  1,
		Documents: map[string]DocumentManifest{
			"keep.md": {ContentHash: contentHash(unchangedContent)},
		},
	}
	var progress []IndexProgress
	_, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "keep.md", Content: unchangedContent},
		{Path: "new.md", Content: "# New\n\nnew body"},
	}}, store, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).
		WithEmbeddingModel("model").
		WithProgress(func(p IndexProgress) { progress = append(progress, p) }).
		BuildAndSave(t.Context(), "notes", "memory.db")
	if err != nil {
		t.Fatalf("BuildAndSave error = %v", err)
	}
	if len(progress) != 2 {
		t.Fatalf("progress calls = %d, want 2", len(progress))
	}
	if !progress[0].Cached {
		t.Fatalf("progress[0].Cached = false, want true for unchanged document")
	}
	if progress[1].Cached {
		t.Fatalf("progress[1].Cached = true, want false for new document")
	}
}

type stubTextSource struct {
	files []TextFile
}

func (s stubTextSource) ReadTextFiles(context.Context, string) ([]TextFile, error) {
	if len(s.files) > 0 {
		return s.files, nil
	}
	return []TextFile{{Path: "note.md", Content: "# Note\nbody"}}, nil
}

type stubIndexStore struct {
	index *Index
}

func (s stubIndexStore) Load(context.Context, string) (*Index, error) {
	if s.index != nil {
		return s.index, nil
	}
	return &Index{}, nil
}

func (stubIndexStore) Save(context.Context, string, *Index) error {
	return nil
}

type capturingIndexStore struct {
	index    *Index
	manifest *IndexManifest
	saved    *Index

	loadCalls int
}

func (s *capturingIndexStore) Load(context.Context, string) (*Index, error) {
	s.loadCalls++
	if s.index != nil {
		return s.index, nil
	}
	return &Index{}, nil
}

func (s *capturingIndexStore) LoadManifest(context.Context, string) (*IndexManifest, error) {
	if s.manifest != nil {
		return s.manifest, nil
	}
	return &IndexManifest{}, nil
}

func (s *capturingIndexStore) Save(_ context.Context, _ string, index *Index) error {
	s.saved = index
	return nil
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

type countingEmbedder struct {
	embedding []float64
	calls     int
}

func (e *countingEmbedder) Embed(context.Context, string) ([]float64, error) {
	e.calls++
	return e.embedding, nil
}

func (e *countingEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float64, error) {
	e.calls++
	out := make([][]float64, len(texts))
	for i := range out {
		out[i] = e.embedding
	}
	return out, nil
}

type fixedTokenizer struct{}

func (fixedTokenizer) TokenizeDocument(_ context.Context, text string) ([]string, error) {
	return strings.Fields(strings.ToLower(text)), nil
}

func (fixedTokenizer) TokenizeQuery(_ context.Context, text string) ([]string, error) {
	return strings.Fields(strings.ToLower(text)), nil
}

type cjkTokenizer struct{}

func (cjkTokenizer) TokenizeDocument(_ context.Context, text string) ([]string, error) {
	tokens := []string{}
	if strings.Contains(text, "東京都") {
		tokens = append(tokens, "東京都", "東京", "都")
	}
	if strings.Contains(text, "京都") && !strings.Contains(text, "東京都") {
		tokens = append(tokens, "京都")
	}
	if strings.Contains(text, "投票作成UI") {
		tokens = append(tokens, "投票作成ui", "投票", "作成", "ui")
	}
	if strings.Contains(text, "選択肢編集") {
		tokens = append(tokens, "選択肢編集", "選択肢", "編集")
	}
	return tokens, nil
}

func (cjkTokenizer) TokenizeQuery(_ context.Context, text string) ([]string, error) {
	switch text {
	case "東京都":
		return []string{"東京都", "東京", "都"}, nil
	case "投票 作成":
		return []string{"投票", "作成"}, nil
	default:
		return strings.Fields(strings.ToLower(text)), nil
	}
}
