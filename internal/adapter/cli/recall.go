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
  anna recall --memory ~/notes/memory.db "search query"

  # Fast lexical search with JSON output
  anna recall --memory ~/notes/memory.db --mode bm25 --json "search query"`,
		RunE: func(cmd *cobra.Command, args []string) error {
			limit := cfg.GetInt("recall.limit")
			jsonOutput := cfg.GetBool("json")
			mode := cfg.GetString("recall.mode")
			ollamaURL := cfg.GetString("ollama-url")
			embeddingModel := cfg.GetString("embedding-model")

			memoryPath := cfg.GetString("memory")
			if memoryPath == "" {
				return fmt.Errorf("memory path is required")
			}
			path, err := defaultMemoryPath(memoryPath)
			if err != nil {
				return err
			}

			searchQuery := strings.TrimSpace(strings.Join(args, " "))
			if searchQuery == "" {
				return fmt.Errorf("query is required")
			}

			searchMode, err := core.ParseSearchMode(mode)
			if err != nil {
				return err
			}
			var embedder core.Embedder
			if searchMode.RequiresEmbedding() && deps.NewEmbedder == nil {
				return fmt.Errorf("embedder factory is required")
			}
			if searchMode.RequiresEmbedding() {
				embedder = deps.NewEmbedder(ollamaURL, embeddingModel)
			}
			tokenizer, err := tokenizerFor(deps)
			if err != nil {
				return err
			}
			searcher := core.NewSearcher(deps.IndexStore, embedder, tokenizer).
				WithEmbeddingModel(embeddingModel)
			results, err := searcher.SearchFile(cmd.Context(), path, searchQuery, limit, searchMode)
			if err != nil {
				if isMemoryNotFound(err) {
					return fmt.Errorf("memory file %q not found; run 'anna nrem <notes-dir>' to create it", path)
				}
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
	cmd.Flags().Int("limit", 10, "maximum results")
	cmd.Flags().String("mode", string(core.SearchModeHybrid), "recall mode: bm25, vector, hybrid, or rrf")
	_ = cfg.BindPFlag("recall.limit", cmd.Flags().Lookup("limit"))
	_ = cfg.BindPFlag("recall.mode", cmd.Flags().Lookup("mode"))
	_ = cmd.RegisterFlagCompletionFunc("mode", completeChoices("bm25", "vector", "hybrid", "rrf"))
	return cmd
}
