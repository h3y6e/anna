package core

import (
	"context"
	"errors"
)

var (
	ErrEmbedTextTooLarge    = errors.New("embedding input exceeds the model's context length")
	ErrIndexVersionMismatch = errors.New("index version does not match this build")
)

type TextSource interface {
	ReadTextFiles(ctx context.Context, source string) ([]TextFile, error)
}

type IndexStore interface {
	Load(ctx context.Context, path string) (*Index, error)
	LoadManifest(ctx context.Context, path string) (*IndexManifest, error)
	// LoadSearchDocuments reads every document with the frequencies of only the given terms, and embeddings when asked.
	LoadSearchDocuments(ctx context.Context, path string, terms []string, withEmbedding bool) (EmbeddingProfile, []Document, error)
	Save(ctx context.Context, path string, index *Index) error
}

type Embedder interface {
	EmbedQuery(ctx context.Context, text string) ([]float64, error)
	EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error)
}

type Tokenizer interface {
	Tokenize(text string) []string
}

type TextFile struct {
	Path    string
	Content string
}
