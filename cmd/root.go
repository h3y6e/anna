package cmd

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/h3y6e/anna/internal/adapter/cli"
	"github.com/h3y6e/anna/internal/adapter/fs"
	"github.com/h3y6e/anna/internal/adapter/openai"
	"github.com/h3y6e/anna/internal/adapter/tokenizer"
	"github.com/h3y6e/anna/internal/core"
)

func Execute(version string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cmd := cli.NewRootCommand(version, cli.Dependencies{
		TextSource: fs.TextSource{},
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(settings cli.EmbedderSettings) (core.Embedder, error) {
			return openai.New(settings.BaseURL, settings.Model, settings.APIKey, settings.QueryPrefix, settings.DocumentPrefix)
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return tokenizer.New()
		},
	})
	cmd.SetContext(ctx)
	return cmd.Execute()
}
