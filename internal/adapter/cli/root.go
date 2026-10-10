package cli

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/h3y6e/anna/internal/core"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

type Dependencies struct {
	TextSource        core.TextSource
	IndexStore        core.IndexStore
	NewEmbedder       func(settings EmbedderSettings) (core.Embedder, error)
	NewTokenizer      func() (core.Tokenizer, error)
	ConfigSearchPaths []string
	ConfigStopDir     string
}

func NewRootCommand(version string, deps Dependencies) *cobra.Command {
	cfg := viper.New()
	cfg.SetEnvPrefix("anna")
	cfg.SetEnvKeyReplacer(strings.NewReplacer(".", "_", "-", "_"))
	cfg.AutomaticEnv()
	var configPath string
	rt := &runtimeConfig{cfg: cfg}
	root := &cobra.Command{
		Use:   "anna",
		Short: "Index and search local text notes",
		Long: `anna turns a directory of text notes into a searchable memory.

		The commands are named after sleep phases:
  nrem     builds an embedding and term index from notes
  recall   searches the memory using bm25, vector, hybrid, or rrf
  settings shows the resolved configuration

Exit codes:
  0  success, help, and version
  1  the command failed
  2  usage error

SIGINT and SIGTERM cancel the command and are raised again.`,
		Version: version,
		Example: `  # Build a memory from a notes directory
  anna nrem ~/notes

  # Search the memory
  anna recall --in ~/notes "search query"

  # Output results as JSON
  anna recall --in ~/notes --json "search query"`,
		SilenceUsage:  true,
		SilenceErrors: true,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			if !cmd.Flags().Changed("config") {
				if env := os.Getenv("ANNA_CONFIG"); env != "" {
					configPath = env
				}
			}
			files, err := readConfig(cfg, configSearchFrom(deps, configPath))
			if err != nil {
				return err
			}
			rt.files = files
			return nil
		},
	}
	commandRuntime.Store(root, rt)
	bindRootFlags(root, cfg, &configPath)

	root.AddCommand(newNREMCommand(cfg, deps))
	root.AddCommand(newRecallCommand(cfg, deps))
	root.AddCommand(newSettingsCommand(cfg, &rt.files))
	return root
}

func bindRootFlags(root *cobra.Command, cfg *viper.Viper, configPath *string) {
	flags := root.PersistentFlags()
	flags.StringVar(configPath, "config", "", "TOML config file path")
	flags.String("memory", defaultMemory, "file name of the memory inside each notes directory")
	flags.BoolP("quiet", "q", false, "suppress progress output")
	flags.Bool("debug", false, "on failure, print a traceback and write a debug log")
	flags.String("embedder-url", "http://localhost:8080", "base URL of an OpenAI-compatible /v1/embeddings endpoint")
	flags.String("embedder-model", "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0", "embedding model")
	flags.String("embedder-query-prefix", "", "text prepended to each search query before embedding")
	flags.String("embedder-document-prefix", "", "text prepended to each note before embedding")
	flags.Bool("json", false, "output results as JSON")
	for _, binding := range []struct{ key, name string }{
		{"memory", "memory"},
		{"quiet", "quiet"},
		{"debug", "debug"},
		{"embedder.url", "embedder-url"},
		{"embedder.model", "embedder-model"},
		{"embedder.query-prefix", "embedder-query-prefix"},
		{"embedder.document-prefix", "embedder-document-prefix"},
		{"json", "json"},
	} {
		_ = cfg.BindPFlag(binding.key, flags.Lookup(binding.name))
	}
}

const defaultMemory = ".anna.db"

type notesScope struct {
	Dir    string
	Memory string
}

func memoryFile(cfg *viper.Viper) (string, error) {
	name := cfg.GetString("memory")
	isEmpty := name == "" || name == "." || name == ".."
	isPath := filepath.Base(name) != name || strings.ContainsAny(name, `/\`)
	if isEmpty || isPath {
		return "", usageFailure(nil, fmt.Sprintf("memory %q must be a file name inside each notes directory", name), nil)
	}
	return name, nil
}

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
				return nil, usageFailure(nil, fmt.Sprintf("notes directories %s and %s overlap; pass directories that do not contain each other", other.Dir, scope.Dir), nil)
			}
		}
		scopes = append(scopes, scope)
	}
	return scopes, nil
}

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
		if env, ok := os.LookupEnv("ANNA_RECALL_IN"); ok {
			dirs = filepath.SplitList(env)
		}
	}
	if len(dirs) == 0 {
		dirs, err = configuredNotes(cfg)
		if err != nil {
			return nil, err
		}
	}
	if len(dirs) == 0 {
		return nil, usageFailure(cmd, "no notes directory to search; pass --in <notes-dir> or set notes in the config", nil)
	}
	return resolveScopes(dirs, name)
}

func requireMemories(scopes []notesScope) error {
	for _, scope := range scopes {
		if _, err := os.Stat(scope.Memory); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return &failure{
					Exit:  1,
					Code:  "not_found",
					Cause: fmt.Sprintf("memory %s not found", scope.Memory),
					Fix:   fmt.Sprintf("anna nrem %s", scope.Dir),
					err:   err,
				}
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
				return nil, configFailure(fmt.Sprintf("notes must be a list of directories, got %v", item), nil)
			}
			dirs = append(dirs, dir)
		}
		return dirs, nil
	default:
		return nil, configFailure(fmt.Sprintf("notes must be a list of directories, got %v", value), nil)
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
