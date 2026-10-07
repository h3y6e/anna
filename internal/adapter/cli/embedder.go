package cli

import (
	"github.com/h3y6e/anna/internal/core"
	"github.com/spf13/viper"
)

// EmbedderSettings selects an OpenAI-compatible /v1/embeddings backend.
type EmbedderSettings struct {
	BaseURL        string
	Model          string
	APIKey         string
	QueryPrefix    string
	DocumentPrefix string
}

func resolveEmbedderSettings(cfg *viper.Viper) EmbedderSettings {
	return EmbedderSettings{
		BaseURL:        cfg.GetString("embedder.url"),
		Model:          cfg.GetString("embedder.model"),
		APIKey:         cfg.GetString("embedder.api-key"),
		QueryPrefix:    cfg.GetString("embedder.query-prefix"),
		DocumentPrefix: cfg.GetString("embedder.document-prefix"),
	}
}

func (s EmbedderSettings) profile() core.EmbeddingProfile {
	return core.EmbeddingProfile{Model: s.Model, QueryPrefix: s.QueryPrefix, DocumentPrefix: s.DocumentPrefix}
}
