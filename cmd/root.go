package cmd

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/h3y6e/anna/internal/adapter/cli"
	"github.com/h3y6e/anna/internal/adapter/fs"
	"github.com/h3y6e/anna/internal/adapter/ollama"
	"github.com/h3y6e/anna/internal/adapter/tokenizer"
	"github.com/h3y6e/anna/internal/core"
)

func Execute(version string) error {
	cli.Version = version
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cmd := cli.NewRootCommand(cli.Dependencies{
		NewTextSource: func() core.TextSource {
			return fs.TextSource{}
		},
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(baseURL string, model string) core.Embedder {
			return ollama.NewEmbedder(baseURL, model)
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return tokenizer.New()
		},
	})
	cmd.SetContext(ctx)
	return cmd.Execute()
}
