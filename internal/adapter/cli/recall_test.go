package cli

import (
	"bytes"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/h3y6e/anna/internal/adapter/fs"
	"github.com/h3y6e/anna/internal/core"
)

func TestRecallUsesConfiguredEmbeddingModel(t *testing.T) {
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

	var capturedModel string
	cmd := NewRootCommand(testDependencies(Dependencies{
		IndexStore: store,
		NewEmbedder: func(_ string, model string) core.Embedder {
			capturedModel = model
			return fixedEmbedder{}
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return fakeTokenizer{}, nil
		},
	}))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetErr(&stderr)
	cmd.SetArgs([]string{"recall", "--memory", memoryPath, "query", "--embedding-model", "qwen3-embedding"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr.String())
	}
	if capturedModel != "qwen3-embedding" {
		t.Fatalf("embedding model = %q, want qwen3-embedding", capturedModel)
	}
}

func TestRecallUsesTOMLConfig(t *testing.T) {
	t.Parallel()

	memoryPath := filepath.Join(t.TempDir(), "memory.db")
	store := fs.IndexStore{}
	if err := store.Save(t.Context(), memoryPath, &core.Index{Version: core.IndexVersion, Documents: []core.Document{
		{
			Path:    "lexical.md",
			Content: "exact keyword match",
			Terms:   map[string]int{"keyword": 1},
			Length:  1,
		},
		{
			Path:    "other.md",
			Content: "different note",
			Terms:   map[string]int{"different": 1},
			Length:  1,
		},
	}}); err != nil {
		t.Fatalf("save fixture memory: %v", err)
	}
	configPath := filepath.Join(t.TempDir(), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf(`memory = %q
json = true

[recall]
mode = "bm25"
limit = 1
`, memoryPath))

	cmd := NewRootCommand(testDependencies(Dependencies{
		IndexStore: store,
		NewTokenizer: func() (core.Tokenizer, error) {
			return fakeTokenizer{}, nil
		},
	}))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetErr(&stderr)
	cmd.SetArgs([]string{"--config", configPath, "recall", "keyword"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"path":"lexical.md"`) {
		t.Fatalf("recall stdout = %q, want JSON lexical result", stdout.String())
	}
	if strings.Contains(stdout.String(), "other.md") {
		t.Fatalf("recall stdout = %q, did not want result beyond configured limit", stdout.String())
	}
}

func TestRecallFlagOverridesTOMLConfig(t *testing.T) {
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
	configPath := filepath.Join(t.TempDir(), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf(`memory = %q

[recall]
mode = "vector"
`, filepath.Join(t.TempDir(), "config-memory.db")))

	cmd := NewRootCommand(testDependencies(Dependencies{
		IndexStore: store,
		NewEmbedder: func(string, string) core.Embedder {
			t.Fatal("flag-selected bm25 mode must not create an embedder")
			return nil
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return fakeTokenizer{}, nil
		},
	}))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetErr(&stderr)
	cmd.SetArgs([]string{"--config", configPath, "recall", "keyword", "--memory", memoryPath, "--mode", "bm25"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "lexical.md") {
		t.Fatalf("recall stdout = %q, want lexical result", stdout.String())
	}
}

func TestRecallBM25ModeDoesNotRequireEmbedder(t *testing.T) {
	t.Parallel()

	memoryPath := filepath.Join(t.TempDir(), "memory.db")
	store := fs.IndexStore{}
	if err := store.Save(t.Context(), memoryPath, &core.Index{Version: core.IndexVersion, Documents: []core.Document{
		{
			Path:    "lexical.md",
			Content: "exact keyword match",
			Terms:   map[string]int{"keyword": 1},
			Length:  1,
		},
		{
			Path:    "other.md",
			Content: "different note",
			Terms:   map[string]int{"different": 1},
			Length:  1,
		},
	}}); err != nil {
		t.Fatalf("save fixture memory: %v", err)
	}

	cmd := NewRootCommand(testDependencies(Dependencies{
		IndexStore: store,
		NewTokenizer: func() (core.Tokenizer, error) {
			return fakeTokenizer{}, nil
		},
	}))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetErr(&stderr)
	cmd.SetArgs([]string{"recall", "--memory", memoryPath, "keyword", "--mode", "bm25"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "lexical.md") {
		t.Fatalf("recall stdout = %q, want lexical result", stdout.String())
	}
	if strings.Contains(stdout.String(), "other.md") {
		t.Fatalf("recall stdout = %q, did not want unrelated result", stdout.String())
	}
}

func TestRecallRejectsUnsupportedMode(t *testing.T) {
	t.Parallel()

	_, _, err := executeCommand("recall", "--memory", "memory.db", "query", "--mode", "unknown")
	if err == nil || !strings.Contains(err.Error(), `unsupported search mode "unknown"`) {
		t.Fatalf("recall mode error = %v, want unsupported mode", err)
	}
}

func TestRecallHelpExposesModeChoice(t *testing.T) {
	t.Parallel()

	stdout, stderr, err := executeCommand("recall", "--help")
	if err != nil {
		t.Fatalf("recall help failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "--mode") {
		t.Fatalf("recall help = %q, want mode flag", stdout)
	}
	if !strings.Contains(stdout, "rrf") {
		t.Fatalf("recall help = %q, want rrf mode", stdout)
	}
}

func TestRecallUsesCWDConfig(t *testing.T) {
	cwd := t.TempDir()
	t.Chdir(cwd)

	memoryPath := filepath.Join(cwd, "memory.db")
	store := fs.IndexStore{}
	if err := store.Save(t.Context(), memoryPath, &core.Index{Version: core.IndexVersion, Documents: []core.Document{{
		Path:    "note.md",
		Content: "exact keyword match",
		Terms:   map[string]int{"keyword": 1},
		Length:  1,
	}}}); err != nil {
		t.Fatalf("save fixture memory: %v", err)
	}
	writeFile(t, filepath.Join(cwd, "anna.toml"), fmt.Sprintf("memory = %q\njson = true\n\n[recall]\nmode = \"bm25\"\n", memoryPath))

	stdout, stderr, err := executeCommand("recall", "keyword")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, `"path":"note.md"`) {
		t.Fatalf("recall stdout = %q, want JSON result", stdout)
	}
}
