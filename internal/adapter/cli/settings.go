package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

type setting struct {
	Key  string
	Flag string
	Env  string
}

func settingsCatalog() []setting {
	return []setting{
		{Key: "memory", Flag: "memory", Env: "ANNA_MEMORY"},
		{Key: "quiet", Flag: "quiet", Env: "ANNA_QUIET"},
		{Key: "json", Flag: "json", Env: "ANNA_JSON"},
		{Key: "debug", Flag: "debug", Env: "ANNA_DEBUG"},
		{Key: "notes", Env: "ANNA_NOTES"},
		{Key: "embedder.url", Flag: "embedder-url", Env: "ANNA_EMBEDDER_URL"},
		{Key: "embedder.model", Flag: "embedder-model", Env: "ANNA_EMBEDDER_MODEL"},
		{Key: "embedder.query-prefix", Flag: "embedder-query-prefix", Env: "ANNA_EMBEDDER_QUERY_PREFIX"},
		{Key: "embedder.document-prefix", Flag: "embedder-document-prefix", Env: "ANNA_EMBEDDER_DOCUMENT_PREFIX"},
		{Key: "embedder.api-key", Env: "ANNA_EMBEDDER_API_KEY"},
		{Key: "nrem.amnesia", Flag: "amnesia", Env: "ANNA_NREM_AMNESIA"},
		{Key: "nrem.dry-run", Flag: "dry-run", Env: "ANNA_NREM_DRY_RUN"},
		{Key: "nrem.yes", Flag: "yes", Env: "ANNA_NREM_YES"},
		{Key: "recall.limit", Flag: "limit", Env: "ANNA_RECALL_LIMIT"},
		{Key: "recall.mode", Flag: "mode", Env: "ANNA_RECALL_MODE"},
		{Key: "recall.in", Flag: "in", Env: "ANNA_RECALL_IN"},
	}
}

func newSettingsCommand(cfg *viper.Viper, files *[]string) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "settings",
		Short: "Show the resolved configuration and where each value came from",
		Long:  "Show each setting, its value, and the origin that supplied it: default, file, env, or flag.\n\nThe config file schema is published at " + docsSchema + ".",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if cfg.GetBool("json") {
				return writeSettingsJSON(cmd.OutOrStdout(), cmd, cfg, *files)
			}
			return writeSettingsText(cmd.OutOrStdout(), cmd, cfg, *files)
		},
	}
	return cmd
}

func writeSettingsText(w io.Writer, cmd *cobra.Command, cfg *viper.Viper, files []string) error {
	defined := definedKeys(files)
	for _, item := range settingsCatalog() {
		value, origin := resolvedSetting(cmd, cfg, defined, item)
		if _, err := fmt.Fprintf(w, "%s\t%s\t%s\n", item.Key, value, origin); err != nil {
			return err
		}
	}
	return nil
}

func writeSettingsJSON(w io.Writer, cmd *cobra.Command, cfg *viper.Viper, files []string) error {
	defined := definedKeys(files)
	enc := json.NewEncoder(w)
	for _, item := range settingsCatalog() {
		value, origin := resolvedSetting(cmd, cfg, defined, item)
		line := map[string]any{
			"record": "setting",
			"schema": outputSchema,
			"key":    item.Key,
			"origin": origin,
			"env":    item.Env,
		}
		if item.Key == "embedder.api-key" && value == "(redacted)" {
			line["redacted"] = true
		} else {
			line["value"] = value
		}
		if err := enc.Encode(line); err != nil {
			return err
		}
	}
	return writeSuccess(w)
}

func resolvedSetting(cmd *cobra.Command, cfg *viper.Viper, defined map[string]bool, item setting) (string, string) {
	value := formatSetting(cfg.Get(item.Key))
	if item.Key == "embedder.api-key" && value != "" {
		value = "(redacted)"
	}
	return value, settingOrigin(cmd, defined, item)
}

func settingOrigin(cmd *cobra.Command, defined map[string]bool, item setting) string {
	if item.Flag != "" && cmd.Flags().Lookup(item.Flag) != nil && cmd.Flags().Changed(item.Flag) {
		return "flag"
	}
	if _, ok := os.LookupEnv(item.Env); ok {
		return "env"
	}
	if defined[item.Key] {
		return "file"
	}
	return "default"
}

func definedKeys(files []string) map[string]bool {
	defined := map[string]bool{}
	for _, path := range files {
		file, err := openConfig(path)
		if err != nil {
			continue
		}
		for _, item := range settingsCatalog() {
			if file.IsSet(item.Key) {
				defined[item.Key] = true
			}
		}
	}
	return defined
}

func formatSetting(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	case []string:
		return strings.Join(v, string(os.PathListSeparator))
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			parts = append(parts, fmt.Sprint(item))
		}
		return strings.Join(parts, string(os.PathListSeparator))
	default:
		return fmt.Sprint(v)
	}
}
