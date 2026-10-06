package cli

import (
	"encoding/json"
	"fmt"

	"github.com/h3y6e/anna/internal/core"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func newREMCommand(cfg *viper.Viper, deps Dependencies) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "rem",
		Short: "Surface related note pairs",
		Long:  "Scan the memory of one notes directory for related, duplicate, or mergeable note pairs and emit them as read-only candidates.",
		Args:  cobra.NoArgs,
		Example: `  # Surface all recombination candidates
  anna rem --in ~/notes

  # Show likely duplicate notes as JSON
  anna rem --in ~/notes --focus echo --json`,
		RunE: func(cmd *cobra.Command, args []string) error {
			jsonOutput := cfg.GetBool("json")
			focus := cfg.GetString("rem.focus")
			limit := cfg.GetInt("rem.limit")
			threshold := cfg.GetFloat64("rem.threshold")

			scope, err := remScope(cmd, cfg)
			if err != nil {
				return err
			}
			if err := requireMemories([]notesScope{scope}); err != nil {
				return err
			}
			options := core.REMOptions{Focus: core.REMFocus(focus), Limit: limit, Threshold: threshold}
			candidates, err := core.NewREMer(deps.IndexStore).REM(cmd.Context(), scope.Memory, options)
			if err != nil {
				return fmt.Errorf("find rem candidates: %w", err)
			}
			for i := range candidates {
				candidates[i].LeftPath = core.ResolveDocumentPath(scope.Memory, candidates[i].LeftPath)
				candidates[i].RightPath = core.ResolveDocumentPath(scope.Memory, candidates[i].RightPath)
			}
			if jsonOutput {
				encoder := json.NewEncoder(cmd.OutOrStdout())
				for _, candidate := range candidates {
					if err := encoder.Encode(candidate); err != nil {
						return fmt.Errorf("write candidate: %w", err)
					}
				}
				return nil
			}

			for _, candidate := range candidates {
				fmt.Fprintf(
					cmd.OutOrStdout(),
					"%s\t%s\t%s\t%.4f\t%s\n",
					candidate.Focus,
					candidate.LeftPath,
					candidate.RightPath,
					candidate.Score,
					candidate.Reason,
				)
			}
			return nil
		},
	}
	cmd.Flags().String("in", "", "notes directory (default: notes from the config)")
	_ = cmd.MarkFlagDirname("in")
	cmd.Flags().String("focus", string(core.REMFocusAll), "candidate focus: echo, synapse, or all")
	cmd.Flags().Int("limit", 10, "maximum candidates")
	cmd.Flags().Float64("threshold", 0.75, "minimum similarity score")
	_ = cfg.BindPFlag("rem.focus", cmd.Flags().Lookup("focus"))
	_ = cfg.BindPFlag("rem.limit", cmd.Flags().Lookup("limit"))
	_ = cfg.BindPFlag("rem.threshold", cmd.Flags().Lookup("threshold"))
	_ = cmd.RegisterFlagCompletionFunc("focus", completeChoices("echo", "synapse", "all"))
	return cmd
}

// remScope resolves the one notes directory rem reads: --in, or the only configured notes directory.
func remScope(cmd *cobra.Command, cfg *viper.Viper) (notesScope, error) {
	name, err := memoryFile(cfg)
	if err != nil {
		return notesScope{}, err
	}
	dir, err := cmd.Flags().GetString("in")
	if err != nil {
		return notesScope{}, err
	}
	if dir == "" {
		dirs, err := configuredNotes(cfg)
		if err != nil {
			return notesScope{}, err
		}
		switch len(dirs) {
		case 0:
			return notesScope{}, fmt.Errorf("no notes directory to scan; pass --in <notes-dir> or set notes in the config")
		case 1:
			dir = dirs[0]
		default:
			return notesScope{}, fmt.Errorf("rem reads one notes directory but the config sets %d; pass --in <notes-dir>", len(dirs))
		}
	}
	return scopeOf(dir, name)
}
