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
		Long:  "Scan the memory database for related, duplicate, or mergeable note pairs and emit them as read-only candidates.",
		Args:  cobra.NoArgs,
		Example: `  # Surface all recombination candidates
  anna rem --memory ~/notes/memory.db

  # Show likely duplicate notes as JSON
  anna rem --memory ~/notes/memory.db --focus echo --json`,
		RunE: func(cmd *cobra.Command, args []string) error {
			jsonOutput := cfg.GetBool("json")
			focus := cfg.GetString("rem.focus")
			limit := cfg.GetInt("rem.limit")
			threshold := cfg.GetFloat64("rem.threshold")

			memoryPath := cfg.GetString("memory")
			if memoryPath == "" {
				return fmt.Errorf("memory path is required")
			}
			path, err := defaultMemoryPath(memoryPath)
			if err != nil {
				return err
			}
			candidates, err := core.NewREMer(deps.IndexStore).REM(cmd.Context(), path, core.REMOptions{
				Focus:     core.REMFocus(focus),
				Limit:     limit,
				Threshold: threshold,
			})
			if err != nil {
				if isMemoryNotFound(err) {
					return fmt.Errorf("memory file %q not found; run 'anna nrem <notes-dir>' to create it", path)
				}
				return fmt.Errorf("find rem candidates: %w", err)
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
	cmd.Flags().String("focus", string(core.REMFocusAll), "candidate focus: echo, synapse, or all")
	cmd.Flags().Int("limit", 10, "maximum candidates")
	cmd.Flags().Float64("threshold", 0.75, "minimum similarity score")
	_ = cfg.BindPFlag("rem.focus", cmd.Flags().Lookup("focus"))
	_ = cfg.BindPFlag("rem.limit", cmd.Flags().Lookup("limit"))
	_ = cfg.BindPFlag("rem.threshold", cmd.Flags().Lookup("threshold"))
	_ = cmd.RegisterFlagCompletionFunc("focus", completeChoices("echo", "synapse", "all"))
	return cmd
}
