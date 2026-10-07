package core

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

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

func (stubIndexStore) LoadManifest(context.Context, string) (*IndexManifest, error) {
	return &IndexManifest{}, nil
}

func (s stubIndexStore) LoadSearchDocuments(context.Context, string, []string, bool) (EmbeddingProfile, []Document, error) {
	if s.index == nil {
		return EmbeddingProfile{}, nil, nil
	}
	return s.index.Embedding, s.index.Documents, nil
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

func (s *capturingIndexStore) LoadSearchDocuments(context.Context, string, []string, bool) (EmbeddingProfile, []Document, error) {
	return s.index.Embedding, s.index.Documents, nil
}

func (s *capturingIndexStore) Save(_ context.Context, _ string, index *Index) error {
	s.saved = index
	return nil
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

type countingEmbedder struct {
	embedding []float64
	calls     int
}

func (e *countingEmbedder) EmbedQuery(context.Context, string) ([]float64, error) {
	e.calls++
	return e.embedding, nil
}

func (e *countingEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	e.calls++
	out := make([][]float64, len(texts))
	for i := range out {
		out[i] = e.embedding
	}
	return out, nil
}

type contextLimitedEmbedder struct {
	maxRunes int

	mu              sync.Mutex
	successfulTexts []string
}

func (e *contextLimitedEmbedder) EmbedQuery(_ context.Context, text string) ([]float64, error) {
	return e.embed(text)
}

func (e *contextLimitedEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i, text := range texts {
		embedding, err := e.embed(text)
		if err != nil {
			return nil, err
		}
		out[i] = embedding
	}
	return out, nil
}

func (e *contextLimitedEmbedder) embed(text string) ([]float64, error) {
	if len([]rune(text)) > e.maxRunes {
		return nil, fmt.Errorf("input too large: %w", ErrEmbedTextTooLarge)
	}
	e.mu.Lock()
	e.successfulTexts = append(e.successfulTexts, text)
	e.mu.Unlock()
	return []float64{float64(len([]rune(text))), 1}, nil
}

type failingEmbedder struct {
	err error
}

func (e failingEmbedder) EmbedQuery(context.Context, string) ([]float64, error) {
	return nil, e.err
}

func (e failingEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	return nil, e.err
}

type fixedTokenizer struct{}

func (fixedTokenizer) Tokenize(text string) []string {
	return strings.Fields(strings.ToLower(text))
}

type cjkTokenizer struct{}

func (cjkTokenizer) Tokenize(text string) []string {
	switch text {
	case "東京都":
		return []string{"東京都", "東京", "都"}
	case "投票 作成":
		return []string{"投票", "作成"}
	}
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
	return tokens
}
