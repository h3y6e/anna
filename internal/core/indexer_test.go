package core

import (
	"errors"
	"fmt"
	"io/fs"
	"slices"
	"strings"
	"testing"
)

func TestWhenSomeNotesAreUnchangedAnIncrementalBuildReusesTheirEmbeddingsAndEmbedsOnlyTheRest(t *testing.T) {
	t.Parallel()

	// Arrange
	unchangedContent := "# Keep\n\nsame body"
	store := &capturingIndexStore{index: &Index{
		Embedding: EmbeddingProfile{Model: "model"},
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
		Embedding:     EmbeddingProfile{Model: "model"},
		DocumentCount: 1,
		Documents: map[string]DocumentManifest{
			"keep.md": {ContentHash: contentHash(unchangedContent)},
		},
	}
	embedder := &countingEmbedder{embedding: []float64{1, 0}}

	// Act
	index, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "keep.md", Content: unchangedContent},
		{Path: "new.md", Content: "# New\n\nnew body"},
	}}, store, embedder, fixedTokenizer{}).WithEmbedding(EmbeddingProfile{Model: "model"}).
		BuildAndSave(t.Context(), "notes", "memory.db", false)

	// Assert
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

func TestWhenNoNoteChangedAnIncrementalBuildNeitherEmbedsNorSaves(t *testing.T) {
	t.Parallel()

	// Arrange
	content := "# Keep\n\nsame body"
	store := &capturingIndexStore{index: &Index{
		Embedding: EmbeddingProfile{Model: "model"},
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
		Embedding:     EmbeddingProfile{Model: "model"},
		DocumentCount: 1,
		Documents: map[string]DocumentManifest{
			"keep.md": {ContentHash: contentHash(content)},
		},
	}
	embedder := &countingEmbedder{embedding: []float64{1, 0}}

	// Act
	index, err := NewIndexer(stubTextSource{files: []TextFile{{Path: "keep.md", Content: content}}}, store, embedder, fixedTokenizer{}).WithEmbedding(EmbeddingProfile{Model: "model"}).
		BuildAndSave(t.Context(), "notes", "memory.db", false)

	// Assert
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
	if got := index.DocumentCount; got != 1 {
		t.Fatalf("index count = %d, want 1", got)
	}
}

func TestWhenSeveralNotesAreBuiltTheyAreEmbeddedInOneBatchCall(t *testing.T) {
	t.Parallel()

	// Arrange
	embedder := &countingEmbedder{embedding: []float64{1, 0}}

	// Act
	index, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "a.md", Content: "# A\n"},
		{Path: "b.md", Content: "# B\n"},
		{Path: "c.md", Content: "# C\n"},
	}}, nil, embedder, fixedTokenizer{}).WithEmbedding(EmbeddingProfile{Model: "model"}).
		build(t.Context(), "notes")

	// Assert
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

func TestWhenARebuildIsRequestedBuildingReembedsEveryNoteDespiteAReusableMemory(t *testing.T) {
	t.Parallel()

	// Arrange
	content := "# Keep\n\nsame body"
	store := &capturingIndexStore{index: &Index{
		Embedding: EmbeddingProfile{Model: "model"},
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

	// Act
	_, err := NewIndexer(stubTextSource{files: []TextFile{{Path: "keep.md", Content: content}}}, store, embedder, fixedTokenizer{}).WithEmbedding(EmbeddingProfile{Model: "model"}).
		BuildAndSave(t.Context(), "notes", "memory.db", true)

	// Assert
	if err != nil {
		t.Fatalf("BuildAndSave error = %v", err)
	}
	if embedder.calls != 1 {
		t.Fatalf("Embed calls = %d, want full rebuild to embed document", embedder.calls)
	}
	if store.saved == nil {
		t.Fatalf("Save was not called for rebuild")
	}
}

func TestWhenAnIgnoredPathIsSetBuildingSkipsOnlyThatFileAtTheSourceRoot(t *testing.T) {
	t.Parallel()

	// Act
	index, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "note.md", Content: "# Note\n"},
		{Path: "memory.md", Content: "binary"},
		{Path: "sub/memory.md", Content: "binary"},
	}}, nil, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).WithIgnoredPath("memory.md").
		build(t.Context(), "notes")

	// Assert
	if err != nil {
		t.Fatalf("Build error = %v", err)
	}
	got := []string{}
	for _, doc := range index.Documents {
		got = append(got, doc.Path)
	}
	slices.Sort(got)
	if want := []string{"note.md", "sub/memory.md"}; !slices.Equal(got, want) {
		t.Fatalf("documents = %v, want %v", got, want)
	}
}

func TestWhenANoteExceedsTheEmbedderContextBuildingSplitsItAndAveragesTheHalves(t *testing.T) {
	t.Parallel()

	// Arrange
	embedder := &contextLimitedEmbedder{maxRunes: 40}
	content := strings.Repeat("ab ", 20) // 60 runes, exceeds maxRunes but splits into two halves that fit

	// Act
	index, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "long.md", Content: content},
	}}, nil, embedder, fixedTokenizer{}).build(t.Context(), "notes")

	// Assert
	if err != nil {
		t.Fatalf("Build error = %v", err)
	}
	if len(index.Documents) != 1 {
		t.Fatalf("document count = %d, want 1", len(index.Documents))
	}
	if len(embedder.successfulTexts) != 2 {
		t.Fatalf("successful embed calls = %d, want 2 from splitting once", len(embedder.successfulTexts))
	}

	first, second := embedder.successfulTexts[0], embedder.successfulTexts[1]
	want := (float64(len([]rune(first))) + float64(len([]rune(second)))) / 2
	got := index.Documents[0].Embedding
	if len(got) != 2 || got[0] != want || got[1] != 1 {
		t.Fatalf("embedding = %#v, want average [%v 1]", got, want)
	}
}

func TestWhenAnOversizedNoteHasNoWhitespaceBuildingSplitsItAtTheMidpoint(t *testing.T) {
	t.Parallel()

	// Arrange
	embedder := &contextLimitedEmbedder{maxRunes: 40}
	content := strings.Repeat("x", 60) // no whitespace anywhere, forces a hard midpoint split

	// Act
	index, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "long.md", Content: content},
	}}, nil, embedder, fixedTokenizer{}).build(t.Context(), "notes")

	// Assert
	if err != nil {
		t.Fatalf("Build error = %v", err)
	}
	if len(embedder.successfulTexts) != 2 {
		t.Fatalf("successful embed calls = %d, want 2 from splitting once", len(embedder.successfulTexts))
	}
	if got := index.Documents[0].Embedding; len(got) != 2 || got[0] != 30 || got[1] != 1 {
		t.Fatalf("embedding = %#v, want [30 1] from averaging two 30-rune halves", got)
	}
}

func TestWhenAnOversizedNoteIsTooShortToSplitBuildingFailsWithErrEmbedTextTooLarge(t *testing.T) {
	t.Parallel()

	// Arrange
	embedder := &contextLimitedEmbedder{maxRunes: 5}
	content := strings.Repeat("x", 30) // exceeds maxRunes but too small to split under the minimum floor

	// Act
	_, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "tiny.md", Content: content},
	}}, nil, embedder, fixedTokenizer{}).build(t.Context(), "notes")

	// Assert
	if err == nil || !errors.Is(err, ErrEmbedTextTooLarge) {
		t.Fatalf("Build error = %v, want error wrapping ErrEmbedTextTooLarge", err)
	}
}

func TestWhenTheEmbedderFailsForAnotherReasonBuildingReturnsThatErrorWithoutSplitting(t *testing.T) {
	t.Parallel()

	// Arrange
	wantErr := errors.New("boom")

	// Act
	_, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "a.md", Content: "short content"},
	}}, nil, failingEmbedder{err: wantErr}, fixedTokenizer{}).build(t.Context(), "notes")

	// Assert
	if err == nil || !errors.Is(err, wantErr) {
		t.Fatalf("Build error = %v, want wrapped %v", err, wantErr)
	}
}

func TestWhenNotesAreBuiltProgressIsReportedOncePerNoteInOrder(t *testing.T) {
	t.Parallel()

	// Arrange
	var progress []IndexProgress

	// Act
	_, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "a.md", Content: "# A\nbody a"},
		{Path: "b.md", Content: "# B\nbody b"},
	}}, nil, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).
		WithProgress(func(p IndexProgress) { progress = append(progress, p) }).
		build(t.Context(), "notes")

	// Assert
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

func TestWhenSomeNotesAreUnchangedAnIncrementalBuildReportsThemAsCached(t *testing.T) {
	t.Parallel()

	// Arrange
	unchangedContent := "# Keep\n\nsame body"
	store := &capturingIndexStore{index: &Index{
		Embedding: EmbeddingProfile{Model: "model"},
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
		Embedding:     EmbeddingProfile{Model: "model"},
		DocumentCount: 1,
		Documents: map[string]DocumentManifest{
			"keep.md": {ContentHash: contentHash(unchangedContent)},
		},
	}
	var progress []IndexProgress

	// Act
	_, err := NewIndexer(stubTextSource{files: []TextFile{
		{Path: "keep.md", Content: unchangedContent},
		{Path: "new.md", Content: "# New\n\nnew body"},
	}}, store, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).
		WithEmbedding(EmbeddingProfile{Model: "model"}).
		WithProgress(func(p IndexProgress) { progress = append(progress, p) }).
		BuildAndSave(t.Context(), "notes", "memory.db", false)

	// Assert
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

func TestWhenAPrefixChangedAnIncrementalBuildReembedsEveryNote(t *testing.T) {
	t.Parallel()

	// Arrange
	content := "# Keep\n\nsame body"
	previous := EmbeddingProfile{Model: "model", QueryPrefix: "query: ", DocumentPrefix: "old: "}
	store := &capturingIndexStore{index: &Index{
		Embedding: previous,
		Documents: []Document{{
			Path:        "keep.md",
			Content:     content,
			ContentHash: contentHash(content),
			Terms:       map[string]int{"keep": 1},
			Length:      1,
			Embedding:   []float64{0, 1},
		}},
	}}
	store.manifest = &IndexManifest{
		Embedding:     previous,
		DocumentCount: 1,
		Documents:     map[string]DocumentManifest{"keep.md": {ContentHash: contentHash(content)}},
	}
	current := EmbeddingProfile{Model: "model", QueryPrefix: "query: ", DocumentPrefix: "new: "}
	embedder := &countingEmbedder{embedding: []float64{1, 0}}

	// Act
	index, err := NewIndexer(stubTextSource{files: []TextFile{{Path: "keep.md", Content: content}}}, store, embedder, fixedTokenizer{}).
		WithEmbedding(current).
		BuildAndSave(t.Context(), "notes", "memory.db", false)

	// Assert
	if err != nil {
		t.Fatalf("BuildAndSave error = %v", err)
	}
	if got := index.Documents[0].Embedding; got[0] != 1 || got[1] != 0 {
		t.Fatalf("embedding = %v, want a fresh embedding", got)
	}
	if store.saved == nil || store.saved.Embedding != current {
		t.Fatalf("saved index = %+v, want embedding profile %+v", store.saved, current)
	}
}

func TestWhenTheMemoryIsMissingOrFromAnotherVersionAnIncrementalBuildStartsFresh(t *testing.T) {
	t.Parallel()

	for name, manifestErr := range map[string]error{
		"missing":          fmt.Errorf("open index: %w", fs.ErrNotExist),
		"version mismatch": fmt.Errorf("index version 4, want 5: %w", ErrIndexVersionMismatch),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Arrange
			store := &capturingIndexStore{manifestErr: manifestErr}

			// Act
			_, err := NewIndexer(stubTextSource{}, store, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).
				BuildAndSave(t.Context(), "notes", "memory.db", false)

			// Assert
			if err != nil {
				t.Fatalf("BuildAndSave error = %v", err)
			}
			if store.saved == nil {
				t.Fatal("Save was not called for a fresh build")
			}
		})
	}
}

func TestWhenTheMemoryIsUnreadableAnIncrementalBuildFailsWithoutOverwritingIt(t *testing.T) {
	t.Parallel()

	// Arrange
	corrupt := errors.New("decode index document count")
	store := &capturingIndexStore{manifestErr: corrupt}

	// Act
	_, err := NewIndexer(stubTextSource{}, store, fixedEmbedder{embedding: []float64{1, 0}}, fixedTokenizer{}).
		BuildAndSave(t.Context(), "notes", "memory.db", false)

	// Assert
	if !errors.Is(err, corrupt) {
		t.Fatalf("BuildAndSave error = %v, want %v", err, corrupt)
	}
	if store.saved != nil {
		t.Fatal("Save was called despite an unreadable memory")
	}
}
