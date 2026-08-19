package cli

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/h3y6e/anna/internal/adapter/fs"
	"github.com/h3y6e/anna/internal/core"
)

func nremWithEmbedderSpy(t *testing.T, extraArgs ...string) (provider, baseURL, model string, err error) {
	t.Helper()
	source := t.TempDir()
	writeFile(t, filepath.Join(source, "note.md"), "note content\n")
	deps := testDependencies(Dependencies{
		NewTextSource: func() core.TextSource { return fs.TextSource{} },
		IndexStore:    fs.IndexStore{},
		NewEmbedder: func(p string, u string, m string) (core.Embedder, error) {
			provider, baseURL, model = p, u, m
			return fakeEmbedder{}, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) { return fakeTokenizer{}, nil },
	})
	args := append([]string{"nrem", source, "--memory", filepath.Join(t.TempDir(), "memory.db")}, extraArgs...)
	_, _, err = executeCommandWithDependencies(deps, args...)
	return provider, baseURL, model, err
}

func TestNREMDefaultsToLlamaCppEmbedder(t *testing.T) {
	t.Parallel()

	provider, baseURL, model, err := nremWithEmbedderSpy(t)
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if provider != "llama.cpp" {
		t.Fatalf("provider = %q, want llama.cpp", provider)
	}
	if baseURL != "http://localhost:8080" {
		t.Fatalf("base URL = %q, want http://localhost:8080", baseURL)
	}
	if model != "ggml-org/embeddinggemma-300M-GGUF:Q8_0" {
		t.Fatalf("model = %q, want ggml-org/embeddinggemma-300M-GGUF:Q8_0", model)
	}
}

func TestNREMSelectsOllamaEmbedderWithProviderDefaults(t *testing.T) {
	t.Parallel()

	provider, baseURL, model, err := nremWithEmbedderSpy(t, "--embedder-provider", "ollama")
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if provider != "ollama" {
		t.Fatalf("provider = %q, want ollama", provider)
	}
	if baseURL != "http://localhost:11434" {
		t.Fatalf("base URL = %q, want http://localhost:11434", baseURL)
	}
	if model != "embeddinggemma" {
		t.Fatalf("model = %q, want embeddinggemma", model)
	}
}

func TestNREMSelectsLlamaCppEmbedderWithProviderDefaults(t *testing.T) {
	t.Parallel()

	provider, baseURL, model, err := nremWithEmbedderSpy(t, "--embedder-provider", "llama.cpp")
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if provider != "llama.cpp" {
		t.Fatalf("provider = %q, want llama.cpp", provider)
	}
	if baseURL != "http://localhost:8080" {
		t.Fatalf("base URL = %q, want http://localhost:8080", baseURL)
	}
	if model != "ggml-org/embeddinggemma-300M-GGUF:Q8_0" {
		t.Fatalf("model = %q, want ggml-org/embeddinggemma-300M-GGUF:Q8_0", model)
	}
}

func TestNREMFlagsOverrideEmbedderProviderDefaults(t *testing.T) {
	t.Parallel()

	provider, baseURL, model, err := nremWithEmbedderSpy(t,
		"--embedder-provider", "llama.cpp",
		"--embedder-url", "http://llama.example:8080",
		"--embedder-model", "custom-model",
	)
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if provider != "llama.cpp" {
		t.Fatalf("provider = %q, want llama.cpp", provider)
	}
	if baseURL != "http://llama.example:8080" {
		t.Fatalf("base URL = %q, want http://llama.example:8080", baseURL)
	}
	if model != "custom-model" {
		t.Fatalf("model = %q, want custom-model", model)
	}
}

func TestNREMRejectsUnsupportedEmbedder(t *testing.T) {
	t.Parallel()

	_, _, _, err := nremWithEmbedderSpy(t, "--embedder-provider", "unknown")
	if err == nil || !strings.Contains(err.Error(), `unsupported embedder "unknown"`) {
		t.Fatalf("nrem embedder error = %v, want unsupported embedder", err)
	}
}

func TestRecallSelectsLlamaCppEmbedderFromTOMLConfig(t *testing.T) {
	t.Parallel()

	memoryPath := filepath.Join(t.TempDir(), "memory.db")
	store := fs.IndexStore{}
	if err := store.Save(t.Context(), memoryPath, &core.Index{Version: core.IndexVersion, Documents: []core.Document{{
		Path:      "note.md",
		Terms:     map[string]int{"query": 1},
		Length:    1,
		Embedding: []float64{1, 0},
	}}}); err != nil {
		t.Fatalf("save fixture memory: %v", err)
	}
	configPath := filepath.Join(t.TempDir(), "anna.toml")
	writeFile(t, configPath, "[embedder]\nprovider = \"llama.cpp\"\n")

	var provider, baseURL string
	deps := testDependencies(Dependencies{
		IndexStore: store,
		NewEmbedder: func(p string, u string, _ string) (core.Embedder, error) {
			provider, baseURL = p, u
			return fixedEmbedder{}, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) { return fakeTokenizer{}, nil },
	})
	_, stderr, err := executeCommandWithDependencies(deps,
		"--config", configPath, "recall", "--memory", memoryPath, "query")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if provider != "llama.cpp" {
		t.Fatalf("provider = %q, want llama.cpp", provider)
	}
	if baseURL != "http://localhost:8080" {
		t.Fatalf("base URL = %q, want http://localhost:8080", baseURL)
	}
}

func TestEmbedderProviderEnvVarSelectsProvider(t *testing.T) {
	t.Setenv("ANNA_EMBEDDER_PROVIDER", "llama.cpp")

	provider, _, _, err := nremWithEmbedderSpy(t)
	if err != nil {
		t.Fatalf("nrem command failed: %v", err)
	}
	if provider != "llama.cpp" {
		t.Fatalf("provider = %q, want llama.cpp from ANNA_EMBEDDER_PROVIDER", provider)
	}
}

func TestRecallBM25ModeIgnoresInvalidEmbedderConfig(t *testing.T) {
	t.Parallel()

	memoryPath := filepath.Join(t.TempDir(), "memory.db")
	store := fs.IndexStore{}
	if err := store.Save(t.Context(), memoryPath, &core.Index{Version: core.IndexVersion, Documents: []core.Document{{
		Path:    "lexical.md",
		Content: "exact keyword match",
		Terms:   map[string]int{"keyword": 1},
		Length:  1,
	}}}); err != nil {
		t.Fatalf("save fixture memory: %v", err)
	}

	deps := testDependencies(Dependencies{
		IndexStore:   store,
		NewTokenizer: func() (core.Tokenizer, error) { return fakeTokenizer{}, nil },
	})
	stdout, stderr, err := executeCommandWithDependencies(deps,
		"recall", "--memory", memoryPath, "keyword", "--mode", "bm25", "--embedder-provider", "bogus")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "lexical.md") {
		t.Fatalf("recall stdout = %q, want lexical result", stdout)
	}
}
