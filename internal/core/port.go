package core

import "context"

type TextSource interface {
	ReadTextFiles(ctx context.Context, source string) ([]TextFile, error)
}

type IndexStore interface {
	Load(ctx context.Context, path string) (*Index, error)
	Save(ctx context.Context, path string, index *Index) error
}

type IndexManifestStore interface {
	LoadManifest(ctx context.Context, path string) (*IndexManifest, error)
}

type SearchIndexStore interface {
	Search(
		ctx context.Context,
		path string,
		query string,
		limit int,
		embedder Embedder,
		tokenizer Tokenizer,
		embeddingModel string,
		mode SearchMode,
	) ([]SearchResult, error)
}

type Embedder interface {
	Embed(ctx context.Context, text string) ([]float64, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float64, error)
}

type Tokenizer interface {
	TokenizeDocument(ctx context.Context, text string) ([]string, error)
	TokenizeQuery(ctx context.Context, text string) ([]string, error)
}

type TextFile struct {
	Path    string
	Content string
}
