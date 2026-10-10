package cli

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/h3y6e/anna/internal/core"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"golang.org/x/term"
)

type nremResult struct {
	Record        string `json:"record"`
	Schema        int    `json:"schema"`
	SourcePath    string `json:"source_path"`
	MemoryPath    string `json:"memory_path"`
	DocumentCount int    `json:"document_count"`
}

func newNREMCommand(cfg *viper.Viper, deps Dependencies) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "nrem [notes-dir]...",
		Short: "Build a search index from notes",
		Long: `Read text files from each notes directory, build an embedding and term index,
and write it to the memory database inside that directory (<notes-dir>/.anna.db by default).
Without arguments, nrem builds the notes directories set in the config.

This command writes a memory. --amnesia is destructive: it overwrites an existing memory.`,
		Args: cobra.ArbitraryArgs,
		Example: `  # Build the memory of a notes directory
  anna nrem ~/notes

  # Build the memories of several notes directories
  anna nrem ~/notes ~/work/notes

  # Forget and rebuild the memory
  anna nrem ~/notes --amnesia --yes`,
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
				return usageFailure(cmd, "no notes directory to index; pass <notes-dir> or set notes in the config", nil)
			}
			scopes, err := resolveScopes(dirs, name)
			if err != nil {
				return err
			}
			if err := confirmAmnesia(cmd, cfg, scopes); err != nil {
				return err
			}
			if cfg.GetBool("nrem.dry-run") {
				return dryRunNREM(cmd, cfg, deps, scopes)
			}
			settings := resolveEmbedderSettings(cfg)
			tokenizer, err := deps.NewTokenizer()
			if err != nil {
				return fmt.Errorf("create tokenizer: %w", err)
			}
			embedder, err := deps.NewEmbedder(settings)
			if err != nil {
				return err
			}
			stderr := cmd.ErrOrStderr()
			indexer := core.NewIndexer(deps.TextSource, deps.IndexStore, embedder, tokenizer).
				WithEmbedding(settings.profile()).
				WithIgnoredPath(name).
				WithProgress(func(p core.IndexProgress) {
					if cfg.GetBool("quiet") {
						return
					}
					if p.Cached {
						fmt.Fprintf(stderr, "  [%d/%d]\t%s\t(cached)\n", p.Current, p.Total, p.Path)
					} else {
						fmt.Fprintf(stderr, "  [%d/%d]\t%s\n", p.Current, p.Total, p.Path)
					}
				})
			jsonOutput := cfg.GetBool("json")
			for _, scope := range scopes {
				if !cfg.GetBool("quiet") {
					fmt.Fprintf(stderr, "nrem\t%s\t%s\tmodel=%s\n", scope.Dir, scope.Memory, settings.Model)
				}
				index, err := indexer.BuildAndSave(cmd.Context(), scope.Dir, scope.Memory, cfg.GetBool("nrem.amnesia"))
				if err != nil {
					return fmt.Errorf("consolidate %s: %w", scope.Dir, err)
				}
				if err := writeNREMResult(cmd.OutOrStdout(), scope, index.DocumentCount, jsonOutput); err != nil {
					return err
				}
			}
			if jsonOutput {
				return writeSuccess(cmd.OutOrStdout())
			}
			return nil
		},
	}
	cmd.Flags().Bool("amnesia", false, "destructive: delete the memory and rebuild it from notes")
	cmd.Flags().Bool("dry-run", false, "print what would be written without writing the memory")
	cmd.Flags().Bool("yes", false, "skip confirmation for a destructive command")
	_ = cfg.BindPFlag("nrem.amnesia", cmd.Flags().Lookup("amnesia"))
	_ = cfg.BindPFlag("nrem.dry-run", cmd.Flags().Lookup("dry-run"))
	_ = cfg.BindPFlag("nrem.yes", cmd.Flags().Lookup("yes"))
	cmd.ValidArgsFunction = func(_ *cobra.Command, _ []string, _ string) ([]string, cobra.ShellCompDirective) {
		return nil, cobra.ShellCompDirectiveFilterDirs
	}
	return cmd
}

func confirmAmnesia(cmd *cobra.Command, cfg *viper.Viper, scopes []notesScope) error {
	if !cfg.GetBool("nrem.amnesia") {
		return nil
	}
	var existing []string
	for _, scope := range scopes {
		if _, err := os.Stat(scope.Memory); err == nil {
			existing = append(existing, scope.Memory)
		} else if !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("check memory %s: %w", scope.Memory, err)
		}
	}
	if len(existing) == 0 {
		return nil
	}
	stderr := cmd.ErrOrStderr()
	for _, path := range existing {
		fmt.Fprintf(stderr, "overwrite\t%s\n", path)
	}
	if cfg.GetBool("nrem.dry-run") || cfg.GetBool("nrem.yes") {
		return nil
	}
	in := cmd.InOrStdin()
	file, ok := in.(*os.File)
	if !ok || !term.IsTerminal(int(file.Fd())) {
		return &failure{
			Exit:  2,
			Code:  "usage",
			Cause: "refusing to overwrite " + strings.Join(existing, ", "),
			Fix:   cmd.CommandPath() + " --yes",
		}
	}
	fmt.Fprint(stderr, "overwrite these memories? [y/N] ")
	line, err := bufio.NewReader(in).ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	answer := strings.ToLower(strings.TrimSpace(line))
	if answer == "y" || answer == "yes" {
		return nil
	}
	return &failure{Exit: 2, Code: "usage", Cause: "overwrite declined", Fix: cmd.CommandPath() + " --yes"}
}

func dryRunNREM(cmd *cobra.Command, cfg *viper.Viper, deps Dependencies, scopes []notesScope) error {
	jsonOutput := cfg.GetBool("json")
	for _, scope := range scopes {
		files, err := deps.TextSource.ReadTextFiles(cmd.Context(), scope.Dir)
		if err != nil {
			return err
		}
		if jsonOutput {
			if err := writeNREMResult(cmd.OutOrStdout(), scope, len(files), true); err != nil {
				return err
			}
			continue
		}
		if _, err := fmt.Fprintf(cmd.OutOrStdout(), "would consolidate %d documents\t%s\n", len(files), scope.Memory); err != nil {
			return err
		}
	}
	if jsonOutput {
		return writeSuccess(cmd.OutOrStdout())
	}
	return nil
}

func writeNREMResult(w io.Writer, scope notesScope, documentCount int, jsonOutput bool) error {
	if !jsonOutput {
		fmt.Fprintf(w, "consolidated %d documents\t%s\n", documentCount, scope.Memory)
		return nil
	}
	if err := json.NewEncoder(w).Encode(nremResult{
		Record:        "index",
		Schema:        outputSchema,
		SourcePath:    scope.Dir,
		MemoryPath:    scope.Memory,
		DocumentCount: documentCount,
	}); err != nil {
		return fmt.Errorf("write result: %w", err)
	}
	return nil
}
