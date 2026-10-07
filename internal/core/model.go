package core

import "time"

const IndexVersion = 5

type EmbeddingProfile struct {
	Model          string
	QueryPrefix    string
	DocumentPrefix string
}

type Index struct {
	Version       int
	Embedding     EmbeddingProfile
	DocumentCount int
	GeneratedAt   time.Time
	Documents     []Document
}

type IndexManifest struct {
	Version       int
	Embedding     EmbeddingProfile
	DocumentCount int
	GeneratedAt   time.Time
	Documents     map[string]DocumentManifest
}

type DocumentManifest struct {
	ContentHash string
}

type Document struct {
	Path        string
	Content     string
	ContentHash string
	Terms       map[string]int
	Length      int
	Embedding   []float64
}

type SearchResult struct {
	Path    string  `json:"path"`
	Score   float64 `json:"score"`
	Snippet string  `json:"snippet"`
}
