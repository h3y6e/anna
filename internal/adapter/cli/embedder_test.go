package cli

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/h3y6e/anna/internal/adapter/fs"
	"github.com/h3y6e/anna/internal/core"
)

func nremWithEmbedderSpy(t *testing.T, extraArgs ...string) (settings EmbedderSettings, err error) {
	t.Helper()
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "note content\n")
	deps := testDependencies(Dependencies{
		NewTextSource: func() core.TextSource { return fs.TextSource{} },
		IndexStore:    fs.IndexStore{},
		NewEmbedder: func(s EmbedderSettings) (core.Embedder, error) {
			settings = s
			return fakeEmbedder{}, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) { return fakeTokenizer{}, nil },
	})
	args := append([]string{"nrem", source}, extraArgs...)
	_, _, err = executeCommandWithDependencies(deps, args...)
	return settings, err
}

func TestNREMDefaultsToLocalLlamaCppEndpoint(t *testing.T) {
	t.Parallel()

	settings, err := nremWithEmbedderSpy(t)
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if settings.BaseURL != "http://localhost:8080" {
		t.Fatalf("base URL = %q, want http://localhost:8080", settings.BaseURL)
	}
	if settings.Model != "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0" {
		t.Fatalf("model = %q, want Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0", settings.Model)
	}
	if settings.APIKey != "" {
		t.Fatalf("API key = %q, want empty", settings.APIKey)
	}
}

func TestNREMFlagsOverrideEmbedderDefaults(t *testing.T) {
	t.Parallel()

	settings, err := nremWithEmbedderSpy(t,
		"--embedder-url", "http://ollama.example:11434",
		"--embedder-model", "qwen3-embedding:0.6b",
	)
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if settings.BaseURL != "http://ollama.example:11434" {
		t.Fatalf("base URL = %q, want http://ollama.example:11434", settings.BaseURL)
	}
	if settings.Model != "qwen3-embedding:0.6b" {
		t.Fatalf("model = %q, want qwen3-embedding:0.6b", settings.Model)
	}
}

func TestEmbedderAPIKeyEnvVarReachesEmbedderFactory(t *testing.T) {
	t.Setenv("ANNA_EMBEDDER_API_KEY", "secret-key")

	settings, err := nremWithEmbedderSpy(t)
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if settings.APIKey != "secret-key" {
		t.Fatalf("API key = %q, want secret-key from ANNA_EMBEDDER_API_KEY", settings.APIKey)
	}
}

func TestEmbedderURLEnvVarOverridesDefault(t *testing.T) {
	t.Setenv("ANNA_EMBEDDER_URL", "http://env.example:8080")

	settings, err := nremWithEmbedderSpy(t)
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if settings.BaseURL != "http://env.example:8080" {
		t.Fatalf("base URL = %q, want http://env.example:8080 from ANNA_EMBEDDER_URL", settings.BaseURL)
	}
}

func TestRecallSelectsEmbedderFromTOMLConfig(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	saveMemory(t, filepath.Join(notes, ".anna.db"), core.Document{
		Path:      "note.md",
		Terms:     map[string]int{"query": 1},
		Length:    1,
		Embedding: []float64{1, 0},
	})
	configPath := filepath.Join(tempDir(t), "anna.toml")
	writeFile(t, configPath, "[embedder]\nurl = \"http://config.example:8080\"\n")

	var settings EmbedderSettings
	deps := testDependencies(Dependencies{
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(s EmbedderSettings) (core.Embedder, error) {
			settings = s
			return fixedEmbedder{}, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) { return fakeTokenizer{}, nil },
	})
	_, stderr, err := executeCommandWithDependencies(deps,
		"--config", configPath, "recall", "--in", notes, "query")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if settings.BaseURL != "http://config.example:8080" {
		t.Fatalf("base URL = %q, want http://config.example:8080", settings.BaseURL)
	}
	if settings.Model != "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0" {
		t.Fatalf("model = %q, want default model", settings.Model)
	}
}

func TestNREMFlagsSetEmbedderPrefixes(t *testing.T) {
	t.Parallel()

	// Act
	settings, err := nremWithEmbedderSpy(t,
		"--embedder-query-prefix", "query: ",
		"--embedder-document-prefix", "passage: ",
	)

	// Assert
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if settings.QueryPrefix != "query: " || settings.DocumentPrefix != "passage: " {
		t.Fatalf("prefixes = %q, %q, want query: and passage: ", settings.QueryPrefix, settings.DocumentPrefix)
	}
}

func TestEmbedderQueryPrefixEnvVarReachesEmbedderFactory(t *testing.T) {
	// Arrange
	t.Setenv("ANNA_EMBEDDER_QUERY_PREFIX", "query: ")

	// Act
	settings, err := nremWithEmbedderSpy(t)

	// Assert
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if settings.QueryPrefix != "query: " {
		t.Fatalf("query prefix = %q, want query: from ANNA_EMBEDDER_QUERY_PREFIX", settings.QueryPrefix)
	}
}

func TestNREMReadsEmbedderPrefixesFromTOMLConfig(t *testing.T) {
	t.Parallel()

	// Arrange
	configPath := filepath.Join(tempDir(t), "anna.toml")
	writeFile(t, configPath, "[embedder]\nquery-prefix = \"Instruct: find notes\\nQuery: \"\ndocument-prefix = \"doc: \"\n")

	// Act
	settings, err := nremWithEmbedderSpy(t, "--config", configPath)

	// Assert
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if settings.QueryPrefix != "Instruct: find notes\nQuery: " || settings.DocumentPrefix != "doc: " {
		t.Fatalf("prefixes = %q, %q, want config values", settings.QueryPrefix, settings.DocumentPrefix)
	}
}

func TestRecallFailsWhenPrefixesDifferFromThoseTheMemoryWasBuiltWith(t *testing.T) {
	t.Parallel()

	// Arrange
	notes := tempDir(t)
	writeFile(t, filepath.Join(notes, "note.md"), "note content\n")
	deps := testDependencies(Dependencies{
		NewTextSource: func() core.TextSource { return fs.TextSource{} },
		IndexStore:    fs.IndexStore{},
		NewEmbedder:   func(EmbedderSettings) (core.Embedder, error) { return fakeEmbedder{}, nil },
		NewTokenizer:  func() (core.Tokenizer, error) { return fakeTokenizer{}, nil },
	})
	if _, stderr, err := executeCommandWithDependencies(deps, "nrem", notes, "--embedder-query-prefix", "query: "); err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
	}

	// Act
	_, _, err := executeCommandWithDependencies(deps, "recall", "--in", notes, "note")

	// Assert
	if err == nil || !strings.Contains(err.Error(), `index was built with query prefix "query: "`) {
		t.Fatalf("recall error = %v, want prefix mismatch", err)
	}
}
