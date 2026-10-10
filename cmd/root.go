package cmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"slices"
	"sync/atomic"
	"syscall"

	"github.com/h3y6e/anna/internal/adapter/cli"
	"github.com/h3y6e/anna/internal/adapter/fs"
	"github.com/h3y6e/anna/internal/adapter/openai"
	"github.com/h3y6e/anna/internal/adapter/tokenizer"
	"github.com/h3y6e/anna/internal/core"
	cobrausage "github.com/jdx/usage/integrations/cobra"
	"github.com/spf13/cobra"
)

func Execute(version string) error {
	root := newRoot(version)
	if slices.Contains(os.Args[1:], "--usage-spec") {
		fmt.Print(cobrausage.Generate(root))
		return nil
	}

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(signals)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var received atomic.Value
	go func() {
		select {
		case sig := <-signals:
			received.Store(sig)
			cancel()
		case <-ctx.Done():
		}
	}()

	root.SetContext(ctx)
	err := cli.Execute(root)
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		if sig, ok := received.Load().(syscall.Signal); ok {
			signal.Stop(signals)
			cli.Reraise(sig)
			os.Exit(128 + int(sig))
		}
	}
	return err
}

func newRoot(version string) *cobra.Command {
	return cli.NewRootCommand(version, cli.Dependencies{
		TextSource: fs.TextSource{},
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(settings cli.EmbedderSettings) (core.Embedder, error) {
			return openai.New(settings.BaseURL, settings.Model, settings.APIKey, settings.QueryPrefix, settings.DocumentPrefix)
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return tokenizer.New()
		},
	})
}
