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
	NewEmbedder       func(settings EmbedderSettings) (core.Embedder, error)
	NewTokenizer      func() (core.Tokenizer, error)
	ConfigSearchPaths []string
}

// EmbedderSettings selects an OpenAI-compatible /v1/embeddings backend.
type EmbedderSettings struct {
	BaseURL        string
	Model          string
	APIKey         string
	QueryPrefix    string
	DocumentPrefix string
}

func resolveEmbedderSettings(cfg *viper.Viper) EmbedderSettings {
	return EmbedderSettings{
		BaseURL:        cfg.GetString("embedder.url"),
		Model:          cfg.GetString("embedder.model"),
		APIKey:         cfg.GetString("embedder.api-key"),
		QueryPrefix:    cfg.GetString("embedder.query-prefix"),
		DocumentPrefix: cfg.GetString("embedder.document-prefix"),
	}
}

func (s EmbedderSettings) profile() core.EmbeddingProfile {
	return core.EmbeddingProfile{Model: s.Model, QueryPrefix: s.QueryPrefix, DocumentPrefix: s.DocumentPrefix}
}

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
  recall searches the memory using bm25, vector, hybrid, or rrf`,
		Version: Version,
		Example: `  # Build a memory from a notes directory
  anna nrem ~/notes

  # Search the memory
  anna recall --in ~/notes "search query"

  # Output results as JSON
  anna recall --in ~/notes --json "search query"`,
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
	root.PersistentFlags().String("memory", defaultMemory, "file name of the memory inside each notes directory")
	root.PersistentFlags().BoolP("quiet", "q", false, "suppress progress output")
	root.PersistentFlags().String("embedder-url", "http://localhost:8080", "base URL of an OpenAI-compatible /v1/embeddings endpoint")
	root.PersistentFlags().String("embedder-model", "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0", "embedding model")
	root.PersistentFlags().String("embedder-query-prefix", "", "text prepended to each search query before embedding")
	root.PersistentFlags().String("embedder-document-prefix", "", "text prepended to each note before embedding")
	root.PersistentFlags().Bool("json", false, "output results as JSON")
	_ = cfg.BindPFlag("memory", root.PersistentFlags().Lookup("memory"))
	_ = cfg.BindPFlag("quiet", root.PersistentFlags().Lookup("quiet"))
	_ = cfg.BindPFlag("embedder.url", root.PersistentFlags().Lookup("embedder-url"))
	_ = cfg.BindPFlag("embedder.model", root.PersistentFlags().Lookup("embedder-model"))
	_ = cfg.BindPFlag("embedder.query-prefix", root.PersistentFlags().Lookup("embedder-query-prefix"))
	_ = cfg.BindPFlag("embedder.document-prefix", root.PersistentFlags().Lookup("embedder-document-prefix"))
	_ = cfg.BindPFlag("json", root.PersistentFlags().Lookup("json"))

	root.AddCommand(newNREMCommand(cfg, deps))
	root.AddCommand(newRecallCommand(cfg, deps))
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
	files, err := readConfigFiles(cfg, configPath, localPaths, globalPaths)
	if err != nil {
		return err
	}
	return resolveConfiguredNotes(cfg, files)
}

// resolveConfiguredNotes resolves relative notes entries against the directory of the config file that sets them.
func resolveConfiguredNotes(cfg *viper.Viper, files []string) error {
	if _, fromEnv := os.LookupEnv("ANNA_NOTES"); fromEnv {
		return nil
	}
	base := ""
	for _, path := range files {
		defines, err := configDefinesNotes(path)
		if err != nil {
			return err
		}
		if defines {
			base = filepath.Dir(path)
		}
	}
	if base == "" {
		return nil
	}
	dirs, err := configuredNotes(cfg)
	if err != nil {
		return err
	}
	resolved := make([]string, 0, len(dirs))
	for _, dir := range dirs {
		expanded, err := expandPath(dir)
		if err != nil {
			return err
		}
		if !filepath.IsAbs(expanded) {
			expanded = filepath.Join(base, expanded)
		}
		resolved = append(resolved, expanded)
	}
	cfg.Set("notes", resolved)
	return nil
}

func configDefinesNotes(path string) (bool, error) {
	file := viper.New()
	file.SetConfigFile(path)
	file.SetConfigType("toml")
	if err := file.ReadInConfig(); err != nil {
		return false, fmt.Errorf("read config %s: %w", path, err)
	}
	return file.IsSet("notes"), nil
}

// readConfigFiles reads the config into cfg and returns the files it read, later files overriding earlier ones.
func readConfigFiles(cfg *viper.Viper, configPath string, localPaths, globalPaths []string) ([]string, error) {
	if configPath != "" {
		path, err := expandPath(configPath)
		if err != nil {
			return nil, err
		}
		cfg.SetConfigFile(path)
		cfg.SetConfigType("toml")
		if err := cfg.ReadInConfig(); err != nil {
			return nil, fmt.Errorf("read config: %w", err)
		}
		return []string{path}, nil
	}

	cfg.SetConfigType("toml")
	var files []string

	if len(globalPaths) > 0 {
		cfg.SetConfigName("config")
		for _, path := range globalPaths {
			cfg.AddConfigPath(path)
		}
		if err := cfg.ReadInConfig(); err == nil {
			files = append(files, cfg.ConfigFileUsed())
		} else if _, ok := errors.AsType[viper.ConfigFileNotFoundError](err); !ok {
			return nil, fmt.Errorf("read global config: %w", err)
		}
	}

	if len(localPaths) > 0 {
		cfg.SetConfigName("anna")
		for _, path := range localPaths {
			cfg.AddConfigPath(path)
		}
		if err := cfg.MergeInConfig(); err == nil {
			files = append(files, cfg.ConfigFileUsed())
		} else if _, ok := errors.AsType[viper.ConfigFileNotFoundError](err); !ok {
			return nil, fmt.Errorf("read local config: %w", err)
		}
	}

	return files, nil
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

const defaultMemory = ".anna.db"

// notesScope is a notes directory and the memory database that lives inside it.
type notesScope struct {
	Dir    string
	Memory string
}

func memoryFile(cfg *viper.Viper) (string, error) {
	name := cfg.GetString("memory")
	if name == "" || name == "." || name == ".." || filepath.Base(name) != name || strings.ContainsAny(name, `/\`) {
		return "", fmt.Errorf("memory %q must be a file name inside each notes directory", name)
	}
	return name, nil
}

// resolveScopes turns notes directories into scopes and rejects directories that are the same or contain each other.
func resolveScopes(dirs []string, name string) ([]notesScope, error) {
	scopes := make([]notesScope, 0, len(dirs))
	for _, dir := range dirs {
		expanded, err := expandPath(dir)
		if err != nil {
			return nil, err
		}
		resolved, err := filepath.EvalSymlinks(expanded)
		if err != nil {
			return nil, fmt.Errorf("notes directory %s: %w", dir, err)
		}
		abs, err := filepath.Abs(resolved)
		if err != nil {
			return nil, fmt.Errorf("resolve %s: %w", dir, err)
		}
		scope := notesScope{Dir: abs, Memory: filepath.Join(abs, name)}
		for _, other := range scopes {
			if overlap(other.Dir, scope.Dir) {
				return nil, fmt.Errorf("notes directories %s and %s overlap; pass directories that do not contain each other", other.Dir, scope.Dir)
			}
		}
		scopes = append(scopes, scope)
	}
	return scopes, nil
}

// searchScopes resolves the directories given with --in, or the configured notes directories.
func searchScopes(cmd *cobra.Command, cfg *viper.Viper) ([]notesScope, error) {
	name, err := memoryFile(cfg)
	if err != nil {
		return nil, err
	}
	dirs, err := cmd.Flags().GetStringArray("in")
	if err != nil {
		return nil, err
	}
	if len(dirs) == 0 {
		dirs, err = configuredNotes(cfg)
		if err != nil {
			return nil, err
		}
	}
	if len(dirs) == 0 {
		return nil, fmt.Errorf("no notes directory to search; pass --in <notes-dir> or set notes in the config")
	}
	return resolveScopes(dirs, name)
}

func requireMemories(scopes []notesScope) error {
	for _, scope := range scopes {
		if _, err := os.Stat(scope.Memory); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return fmt.Errorf("memory %s not found; run 'anna nrem %s' to create it", scope.Memory, scope.Dir)
			}
			return fmt.Errorf("check memory %s: %w", scope.Memory, err)
		}
	}
	return nil
}

func configuredNotes(cfg *viper.Viper) ([]string, error) {
	switch value := cfg.Get("notes").(type) {
	case nil:
		return nil, nil
	case string:
		return filepath.SplitList(value), nil
	case []string:
		return value, nil
	case []any:
		dirs := make([]string, 0, len(value))
		for _, item := range value {
			dir, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("notes must be a list of directories, got %v", item)
			}
			dirs = append(dirs, dir)
		}
		return dirs, nil
	default:
		return nil, fmt.Errorf("notes must be a list of directories, got %v", value)
	}
}

func overlap(a string, b string) bool {
	if contains(a, b) || contains(b, a) {
		return true
	}
	infoA, errA := os.Stat(a)
	infoB, errB := os.Stat(b)
	return errA == nil && errB == nil && os.SameFile(infoA, infoB)
}

func contains(dir string, path string) bool {
	rel, err := filepath.Rel(dir, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func addInFlag(cmd *cobra.Command) {
	cmd.Flags().StringArray("in", nil, "notes directory to read; repeat for several (default: notes from the config)")
	_ = cmd.MarkFlagDirname("in")
}

func completeChoices(choices ...string) func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective) {
	return func(_ *cobra.Command, _ []string, _ string) ([]string, cobra.ShellCompDirective) {
		return choices, cobra.ShellCompDirectiveNoFileComp
	}
}
