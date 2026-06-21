package cli

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/h3y6e/anna/internal/core"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

type Dependencies struct {
	NewTextSource     func() core.TextSource
	IndexStore        core.IndexStore
	NewEmbedder       func(baseURL string, model string) core.Embedder
	NewTokenizer      func() (core.Tokenizer, error)
	ConfigSearchPaths []string
}

const defaultEmbeddingModel = "embeddinggemma"

func NewRootCommand(deps Dependencies) *cobra.Command {
	cfg := viper.New()
	cfg.SetEnvPrefix("anna")
	cfg.SetEnvKeyReplacer(strings.NewReplacer(".", "_", "-", "_"))
	cfg.AutomaticEnv()
	var configPath string
	root := &cobra.Command{
		Use:   "anna",
		Short: "Index and search local text notes",
		Long: `anna turns a directory of text notes into a searchable memory.

The commands are named after sleep phases:
  nrem   builds an embedding and term index from notes
  recall searches the index using bm25, vector, hybrid, or rrf
  rem    surfaces related note pairs`,
		Version: Version,
		Example: `  # Build a memory from a notes directory
  anna nrem ~/notes

  # Search the memory
  anna recall --memory ~/notes/memory.db "search query"

  # Output results as JSON
  anna recall --memory ~/notes/memory.db --json "search query"`,
		SilenceUsage:  true,
		SilenceErrors: true,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			localPaths, globalPaths := buildConfigSearchPaths(deps)
			if err := readConfig(cfg, configPath, localPaths, globalPaths); err != nil {
				return err
			}
			if cfg.GetBool("quiet") {
				cmd.SetErr(io.Discard)
			}
			return nil
		},
	}
	root.PersistentFlags().StringVar(&configPath, "config", "", "TOML config file path")
	root.PersistentFlags().StringP("memory", "m", "", "memory database path")
	root.PersistentFlags().BoolP("quiet", "q", false, "suppress progress output")
	root.PersistentFlags().String("ollama-url", "http://localhost:11434", "Ollama base URL")
	root.PersistentFlags().String("embedding-model", defaultEmbeddingModel, "Ollama embedding model")
	root.PersistentFlags().Bool("json", false, "output results as JSON")
	_ = cfg.BindPFlag("memory", root.PersistentFlags().Lookup("memory"))
	_ = cfg.BindPFlag("quiet", root.PersistentFlags().Lookup("quiet"))
	_ = cfg.BindPFlag("ollama-url", root.PersistentFlags().Lookup("ollama-url"))
	_ = cfg.BindPFlag("embedding-model", root.PersistentFlags().Lookup("embedding-model"))
	_ = cfg.BindPFlag("json", root.PersistentFlags().Lookup("json"))

	root.AddCommand(newNREMCommand(cfg, deps))
	root.AddCommand(newRecallCommand(cfg, deps))
	root.AddCommand(newREMCommand(cfg, deps))
	root.AddCommand(newVersionCommand())
	return root
}

func buildConfigSearchPaths(deps Dependencies) (localPaths []string, globalPaths []string) {
	if cwd, err := os.Getwd(); err == nil && cwd != "" {
		localPaths = append(localPaths, cwd)
	}

	if deps.ConfigSearchPaths != nil {
		globalPaths = append(globalPaths, deps.ConfigSearchPaths...)
	} else {
		globalPaths = append(globalPaths, defaultConfigSearchPaths()...)
	}
	return localPaths, globalPaths
}

func readConfig(cfg *viper.Viper, configPath string, localPaths, globalPaths []string) error {
	if configPath != "" {
		path, err := expandPath(configPath)
		if err != nil {
			return err
		}
		cfg.SetConfigFile(path)
		cfg.SetConfigType("toml")
		if err := cfg.ReadInConfig(); err != nil {
			return fmt.Errorf("read config: %w", err)
		}
		return nil
	}

	cfg.SetConfigType("toml")

	if len(globalPaths) > 0 {
		cfg.SetConfigName("config")
		for _, path := range globalPaths {
			cfg.AddConfigPath(path)
		}
		if err := cfg.ReadInConfig(); err != nil {
			if _, ok := errors.AsType[viper.ConfigFileNotFoundError](err); !ok {
				return fmt.Errorf("read global config: %w", err)
			}
		}
	}

	if len(localPaths) > 0 {
		cfg.SetConfigName("anna")
		for _, path := range localPaths {
			cfg.AddConfigPath(path)
		}
		if err := cfg.MergeInConfig(); err != nil {
			if _, ok := errors.AsType[viper.ConfigFileNotFoundError](err); !ok {
				return fmt.Errorf("read local config: %w", err)
			}
		}
	}

	return nil
}

func defaultConfigSearchPaths() []string {
	paths := make([]string, 0, 2)
	if xdgConfigHome := os.Getenv("XDG_CONFIG_HOME"); xdgConfigHome != "" {
		if path, err := expandPath(xdgConfigHome); err == nil {
			paths = append(paths, filepath.Join(path, "anna"))
		}
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		path := filepath.Join(home, ".config", "anna")
		if len(paths) == 0 || paths[len(paths)-1] != path {
			paths = append(paths, path)
		}
	}
	return paths
}

func tokenizerFor(deps Dependencies) (core.Tokenizer, error) {
	if deps.NewTokenizer == nil {
		return nil, fmt.Errorf("tokenizer factory is required")
	}
	tokenizer, err := deps.NewTokenizer()
	if err != nil {
		return nil, fmt.Errorf("create tokenizer: %w", err)
	}
	return tokenizer, nil
}

func expandPath(path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	if path == "~" || strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home directory: %w", err)
		}
		if path == "~" {
			return home, nil
		}
		return filepath.Join(home, strings.TrimPrefix(path, "~/")), nil
	}
	return path, nil
}

func defaultNREMMemoryPath(memoryPath string, sourcePath string) (string, error) {
	if memoryPath != "" {
		return expandPath(memoryPath)
	}
	return filepath.Join(sourcePath, "memory.db"), nil
}

func defaultMemoryPath(memoryPath string) (string, error) {
	if memoryPath != "" {
		return expandPath(memoryPath)
	}
	return "memory.db", nil
}

func completeChoices(choices ...string) func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective) {
	return func(_ *cobra.Command, _ []string, _ string) ([]string, cobra.ShellCompDirective) {
		return choices, cobra.ShellCompDirectiveNoFileComp
	}
}

func isMemoryNotFound(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, os.ErrNotExist) {
		return true
	}
	return strings.Contains(err.Error(), "no such file") || strings.Contains(err.Error(), "system cannot find")
}
