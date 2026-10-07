package cli

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/h3y6e/anna/internal/core"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func newRecallCommand(cfg *viper.Viper, deps Dependencies) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "recall [query]",
		Short: "Search the memory database",
		Long:  "Search the memory database for the given query and return the most relevant notes.",
		Args:  cobra.ArbitraryArgs,
		Example: `  # Search with the default hybrid mode
  anna recall --in ~/notes "search query"

  # Search several notes directories as one memory
  anna recall --in ~/notes --in ~/work/notes "search query"

  # Fast lexical search with JSON output
  anna recall --in ~/notes --mode bm25 --json "search query"`,
		RunE: func(cmd *cobra.Command, args []string) error {
			limit := cfg.GetInt("recall.limit")
			jsonOutput := cfg.GetBool("json")
			mode := cfg.GetString("recall.mode")

			scopes, err := searchScopes(cmd, cfg)
			if err != nil {
				return err
			}

			searchQuery := strings.TrimSpace(strings.Join(args, " "))
			if searchQuery == "" {
				return fmt.Errorf("query is required")
			}

			if limit < 1 {
				return fmt.Errorf("limit must be at least 1, got %d", limit)
			}

			searchMode, err := core.ParseSearchMode(mode)
			if err != nil {
				return err
			}
			if err := requireMemories(scopes); err != nil {
				return err
			}
			var embedder core.Embedder
			var settings EmbedderSettings
			if searchMode.RequiresEmbedding() {
				settings = resolveEmbedderSettings(cfg)
				embedder, err = deps.NewEmbedder(settings)
				if err != nil {
					return err
				}
			}
			tokenizer, err := deps.NewTokenizer()
			if err != nil {
				return fmt.Errorf("create tokenizer: %w", err)
			}
			searcher := core.NewSearcher(deps.IndexStore, embedder, tokenizer).
				WithEmbedding(settings.profile())
			memories := make([]string, len(scopes))
			for i, scope := range scopes {
				memories[i] = scope.Memory
			}
			results, err := searcher.SearchFiles(cmd.Context(), memories, searchQuery, limit, searchMode)
			if err != nil {
				return fmt.Errorf("search memory: %w", err)
			}
			if jsonOutput {
				encoder := json.NewEncoder(cmd.OutOrStdout())
				for _, result := range results {
					if err := encoder.Encode(result); err != nil {
						return fmt.Errorf("write result: %w", err)
					}
				}
				return nil
			}

			for _, result := range results {
				fmt.Fprintf(cmd.OutOrStdout(), "%s\t%.4f\t%s\n", result.Path, result.Score, result.Snippet)
			}
			return nil
		},
	}
	addInFlag(cmd)
	cmd.Flags().Int("limit", 10, "maximum results")
	cmd.Flags().String("mode", string(core.SearchModeHybrid), "recall mode: bm25, vector, hybrid, or rrf")
	_ = cfg.BindPFlag("recall.limit", cmd.Flags().Lookup("limit"))
	_ = cfg.BindPFlag("recall.mode", cmd.Flags().Lookup("mode"))
	_ = cmd.RegisterFlagCompletionFunc("mode", completeChoices("bm25", "vector", "hybrid", "rrf"))
	return cmd
}
