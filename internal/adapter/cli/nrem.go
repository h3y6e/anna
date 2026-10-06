package cli

import (
	"encoding/json"
	"fmt"
	"path/filepath"

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
		Use:   "nrem <notes-dir>...",
		Short: "Build a search index from notes",
		Long: `Read text files from each <notes-dir>, build an embedding and term index,
and write it to the memory database inside that directory (<notes-dir>/.anna.db by default).
Without arguments, nrem builds the notes directories set in the config.`,
		Args: cobra.ArbitraryArgs,
		Example: `  # Build the memory of a notes directory
  anna nrem ~/notes

  # Build the memories of several notes directories
  anna nrem ~/notes ~/work/notes

  # Forget and rebuild the memory
  anna nrem ~/notes --amnesia`,
		RunE: func(cmd *cobra.Command, args []string) error {
			name, err := memoryFile(cfg)
			if err != nil {
				return err
			}
			dirs := args
			if len(dirs) == 0 {
				dirs, err = configuredNotes(cfg)
				if err != nil {
					return err
				}
			}
			if len(dirs) == 0 {
				return fmt.Errorf("no notes directory to index; pass <notes-dir> or set notes in the config")
			}
			scopes, err := resolveScopes(dirs, name)
			if err != nil {
				return err
			}
			amnesia := cfg.GetBool("nrem.amnesia")
			jsonOutput := cfg.GetBool("json")

			if deps.NewEmbedder == nil {
				return fmt.Errorf("embedder factory is required")
			}
			settings := resolveEmbedderSettings(cfg)
			tokenizer, err := tokenizerFor(deps)
			if err != nil {
				return err
			}
			embedder, err := deps.NewEmbedder(settings)
			if err != nil {
				return err
			}
			for _, scope := range scopes {
				if err := consolidate(cmd, deps, tokenizer, embedder, settings.Model, scope, amnesia, jsonOutput); err != nil {
					return err
				}
			}
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

func consolidate(
	cmd *cobra.Command,
	deps Dependencies,
	tokenizer core.Tokenizer,
	embedder core.Embedder,
	model string,
	scope notesScope,
	amnesia bool,
	jsonOutput bool,
) error {
	w := cmd.ErrOrStderr()
	fmt.Fprintf(w, "nrem\t%s\t%s\tmodel=%s\n", scope.Dir, scope.Memory, model)
	var source core.TextSource
	if deps.NewTextSource != nil {
		source = deps.NewTextSource()
	}
	indexer := core.NewIndexer(source, deps.IndexStore, embedder, tokenizer).
		WithEmbeddingModel(model).
		WithIgnoredPath(filepath.Base(scope.Memory)).
		WithProgress(func(p core.IndexProgress) {
			if p.Cached {
				fmt.Fprintf(w, "  [%d/%d]\t%s\t(cached)\n", p.Current, p.Total, p.Path)
			} else {
				fmt.Fprintf(w, "  [%d/%d]\t%s\n", p.Current, p.Total, p.Path)
			}
		})
	index, err := indexer.BuildAndSaveWithOptions(cmd.Context(), scope.Dir, scope.Memory, core.IndexBuildOptions{Rebuild: amnesia})
	if err != nil {
		return fmt.Errorf("consolidate %s: %w", scope.Dir, err)
	}
	if jsonOutput {
		if err := json.NewEncoder(cmd.OutOrStdout()).Encode(nremResult{
			SourcePath:    scope.Dir,
			MemoryPath:    scope.Memory,
			DocumentCount: index.Count(),
		}); err != nil {
			return fmt.Errorf("write result: %w", err)
		}
		return nil
	}
	fmt.Fprintf(cmd.OutOrStdout(), "consolidated %d documents\t%s\n", index.Count(), scope.Memory)
	return nil
}
