package core

import "time"

const IndexVersion = 5

type Index struct {
	Version        int        `json:"version"`
	Source         string     `json:"source"`
	EmbeddingModel string     `json:"embedding_model,omitempty"`
	DocumentCount  int        `json:"document_count,omitempty"`
	GeneratedAt    time.Time  `json:"generated_at"`
	Documents      []Document `json:"documents"`
}

func (i *Index) Count() int {
	if i == nil {
		return 0
	}
	if i.DocumentCount > 0 {
		return i.DocumentCount
	}
	return len(i.Documents)
}

type IndexManifest struct {
	Version        int
	Source         string
	EmbeddingModel string
	DocumentCount  int
	GeneratedAt    time.Time
	Documents      map[string]DocumentManifest
}

type DocumentManifest struct {
	ContentHash string
}

type Document struct {
	Path        string         `json:"path"`
	Content     string         `json:"content"`
	ContentHash string         `json:"content_hash,omitempty"`
	Terms       map[string]int `json:"terms"`
	Length      int            `json:"length"`
	Embedding   []float64      `json:"embedding,omitempty"`
}

type SearchResult struct {
	Path    string  `json:"path"`
	Score   float64 `json:"score"`
	Snippet string  `json:"snippet"`
}
