package cli

import (
	"encoding/json"
	"fmt"

	"github.com/h3y6e/anna/internal/core"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

type nremResult struct {
	SourcePath    string `json:"source_path"`
	MemoryPath    string `json:"memory_path"`
	DocumentCount int    `json:"document_count"`
}

func newNREMCommand(cfg *viper.Viper, deps Dependencies) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "nrem <notes-dir>",
		Short: "Build a search index from notes",
		Long: `Read text files from <notes-dir>, build an embedding and term index,
and write it to the memory database. Defaults to <notes-dir>/memory.db.`,
		Args: cobra.ExactArgs(1),
		Example: `  # Build the memory index next to the source directory
  anna nrem ~/notes

  # Rebuild memory at a custom path
  anna nrem ~/notes --memory ~/notes/memory.db --amnesia`,
		RunE: func(cmd *cobra.Command, args []string) error {
			sourcePath, err := expandPath(args[0])
			if err != nil {
				return err
			}
			memoryPath := cfg.GetString("memory")
			outputPath, err := defaultNREMMemoryPath(memoryPath, sourcePath)
			if err != nil {
				return err
			}
			ollamaURL := cfg.GetString("ollama-url")
			embeddingModel := cfg.GetString("embedding-model")
			amnesia := cfg.GetBool("nrem.amnesia")
			jsonOutput := cfg.GetBool("json")

			if deps.NewEmbedder == nil {
				return fmt.Errorf("embedder factory is required")
			}
			tokenizer, err := tokenizerFor(deps)
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.ErrOrStderr(), "nrem\t%s\t%s\tmodel=%s\n", sourcePath, outputPath, embeddingModel)
			w := cmd.ErrOrStderr()
			var source core.TextSource
			if deps.NewTextSource != nil {
				source = deps.NewTextSource()
			}
			embedder := deps.NewEmbedder(ollamaURL, embeddingModel)
			indexer := core.NewIndexer(source, deps.IndexStore, embedder, tokenizer).
				WithEmbeddingModel(embeddingModel).
				WithProgress(func(p core.IndexProgress) {
					if p.Cached {
						fmt.Fprintf(w, "  [%d/%d]\t%s\t(cached)\n", p.Current, p.Total, p.Path)
					} else {
						fmt.Fprintf(w, "  [%d/%d]\t%s\n", p.Current, p.Total, p.Path)
					}
				})
			index, err := indexer.BuildAndSaveWithOptions(
				cmd.Context(),
				sourcePath,
				outputPath,
				core.IndexBuildOptions{Rebuild: amnesia},
			)
			if err != nil {
				return fmt.Errorf("consolidate notes: %w", err)
			}
			if jsonOutput {
				encoder := json.NewEncoder(cmd.OutOrStdout())
				if err := encoder.Encode(nremResult{
					SourcePath:    sourcePath,
					MemoryPath:    outputPath,
					DocumentCount: index.Count(),
				}); err != nil {
					return fmt.Errorf("write result: %w", err)
				}
				return nil
			}
			fmt.Fprintf(cmd.OutOrStdout(), "consolidated %d documents\t%s\n", index.Count(), outputPath)
			return nil
		},
	}
	cmd.Flags().Bool("amnesia", false, "forget existing memory and rebuild it from notes")
	_ = cfg.BindPFlag("nrem.amnesia", cmd.Flags().Lookup("amnesia"))
	cmd.ValidArgsFunction = func(_ *cobra.Command, _ []string, _ string) ([]string, cobra.ShellCompDirective) {
		return nil, cobra.ShellCompDirectiveFilterDirs
	}
	return cmd
}
