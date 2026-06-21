package cli

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/h3y6e/anna/internal/adapter/fs"
	"github.com/h3y6e/anna/internal/core"
)

func TestNREMDefaultsToNotesMemoryDB(t *testing.T) {
	t.Parallel()

	source := t.TempDir()
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nDefault memory path lives next to source files.\n")
	memoryPath := filepath.Join(source, "memory.db")

	stdout, stderr, err := executeCommand("nrem", source)
	if err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, memoryPath) {
		t.Fatalf("nrem stdout = %q, want default memory path %q", stdout, memoryPath)
	}
	if _, err := os.Stat(memoryPath); err != nil {
		t.Fatalf("default memory file was not created at %s: %v", memoryPath, err)
	}
}

func TestNREMWritesProgressToStderr(t *testing.T) {
	t.Parallel()

	source := t.TempDir()
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nProgress should be visible during NREM consolidation.\n")
	memoryPath := filepath.Join(t.TempDir(), "memory.db")

	_, stderr, err := executeCommand("nrem", source, "--memory", memoryPath)
	if err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stderr, "nrem\t") {
		t.Fatalf("nrem stderr = %q, want nrem progress", stderr)
	}
	if !strings.Contains(stderr, source) {
		t.Fatalf("nrem stderr = %q, want source path %q", stderr, source)
	}
	if !strings.Contains(stderr, memoryPath) {
		t.Fatalf("nrem stderr = %q, want memory path %q", stderr, memoryPath)
	}
	if !strings.Contains(stderr, "model=embeddinggemma") {
		t.Fatalf("nrem stderr = %q, want embedding model", stderr)
	}
	if !strings.Contains(stderr, "[1/1]") {
		t.Fatalf("nrem stderr = %q, want per-document progress", stderr)
	}
	if !strings.Contains(stderr, "note.md") {
		t.Fatalf("nrem stderr = %q, want document path in progress", stderr)
	}
}

func TestNREMQuietSuppressesStderr(t *testing.T) {
	t.Parallel()

	source := t.TempDir()
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nQuiet mode should suppress progress.\n")
	memoryPath := filepath.Join(t.TempDir(), "memory.db")

	stdout, stderr, err := executeCommand("nrem", source, "--memory", memoryPath, "--quiet")
	if err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
	}
	if stderr != "" {
		t.Fatalf("nrem stderr = %q, want empty with --quiet", stderr)
	}
	if !strings.Contains(stdout, "consolidated") {
		t.Fatalf("nrem stdout = %q, want consolidated output even with --quiet", stdout)
	}
}

func TestNREMHelpExposesEmbeddingModelChoice(t *testing.T) {
	t.Parallel()

	stdout, stderr, err := executeCommand("nrem", "--help")
	if err != nil {
		t.Fatalf("nrem help failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "--embedding-model") {
		t.Fatalf("nrem help = %q, want embedding model choice", stdout)
	}
	if strings.Contains(stdout, "--embed ") {
		t.Fatalf("nrem help = %q, did not want optional embedding flag", stdout)
	}
}

func TestNREMHelpExposesAmnesiaChoice(t *testing.T) {
	t.Parallel()

	stdout, stderr, err := executeCommand("nrem", "--help")
	if err != nil {
		t.Fatalf("nrem help failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "--amnesia") {
		t.Fatalf("nrem help = %q, want amnesia flag", stdout)
	}
}

func TestNREMOutputsJSON(t *testing.T) {
	t.Parallel()

	source := t.TempDir()
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nJSON output should include source and memory paths.\n")
	memoryPath := filepath.Join(t.TempDir(), "memory.db")

	stdout, stderr, err := executeCommand("nrem", source, "--memory", memoryPath, "--json")
	if err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, `"source_path":`) {
		t.Fatalf("nrem stdout = %q, want source_path", stdout)
	}
	if !strings.Contains(stdout, `"memory_path":`) {
		t.Fatalf("nrem stdout = %q, want memory_path", stdout)
	}
	if !strings.Contains(stdout, `"document_count":`) {
		t.Fatalf("nrem stdout = %q, want document_count", stdout)
	}
	if strings.Contains(stdout, "consolidated") {
		t.Fatalf("nrem stdout = %q, did not want plain text summary with --json", stdout)
	}
}

func TestNREMUsesConfiguredEmbedder(t *testing.T) {
	t.Parallel()

	source := t.TempDir()
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nEnglish and 日本語 notes.\n")
	var capturedBaseURL string
	var capturedModel string
	cmd := NewRootCommand(testDependencies(Dependencies{
		NewTextSource: func() core.TextSource {
			return fs.TextSource{}
		},
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(baseURL string, model string) core.Embedder {
			capturedBaseURL = baseURL
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
	cmd.SetArgs([]string{"nrem", source})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr.String())
	}
	if capturedBaseURL != "http://localhost:11434" {
		t.Fatalf("default Ollama URL = %q, want http://localhost:11434", capturedBaseURL)
	}
	if capturedModel != "embeddinggemma" {
		t.Fatalf("default embedding model = %q, want embeddinggemma", capturedModel)
	}
}

func TestNREMUsesTokenizerFactory(t *testing.T) {
	t.Parallel()

	source := t.TempDir()
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nEnglish and 日本語 notes.\n")
	var called bool
	cmd := NewRootCommand(testDependencies(Dependencies{
		NewTextSource: func() core.TextSource {
			return fs.TextSource{}
		},
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(string, string) core.Embedder {
			return fixedEmbedder{}
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			called = true
			return fakeTokenizer{}, nil
		},
	}))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetErr(&stderr)
	cmd.SetArgs([]string{"nrem", source})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr.String())
	}
	if !called {
		t.Fatal("tokenizer factory was not called")
	}
}

func TestNREMUsesConfiguredEmbeddingModel(t *testing.T) {
	t.Parallel()

	source := t.TempDir()
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nEnglish and 日本語 notes.\n")
	var capturedModel string
	cmd := NewRootCommand(testDependencies(Dependencies{
		NewTextSource: func() core.TextSource {
			return fs.TextSource{}
		},
		IndexStore: fs.IndexStore{},
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
	cmd.SetArgs([]string{"nrem", source, "--embedding-model", "qwen3-embedding"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr.String())
	}
	if capturedModel != "qwen3-embedding" {
		t.Fatalf("embedding model = %q, want qwen3-embedding", capturedModel)
	}
}

func TestNREMUsesTOMLConfig(t *testing.T) {
	t.Parallel()

	source := t.TempDir()
	writeFile(t, filepath.Join(source, "keep.md"), "# Keep\n\nvisible note\n")
	if err := os.Mkdir(filepath.Join(source, "skip"), 0o700); err != nil {
		t.Fatalf("mkdir skip fixture: %v", err)
	}
	writeFile(t, filepath.Join(source, "skip", "hidden.md"), "# Hidden\n\nthis should now be indexed\n")
	memoryPath := filepath.Join(t.TempDir(), "configured.db")
	configPath := filepath.Join(t.TempDir(), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf(`memory = %q
ollama-url = "http://ollama.example"
embedding-model = "qwen3-embedding"

[nrem]
amnesia = false
`, memoryPath))

	var capturedBaseURL string
	var capturedModel string
	cmd := NewRootCommand(testDependencies(Dependencies{
		NewTextSource: func() core.TextSource {
			return fs.TextSource{}
		},
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(baseURL string, model string) core.Embedder {
			capturedBaseURL = baseURL
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
	cmd.SetArgs([]string{"--config", configPath, "nrem", source})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "consolidated 2 documents") {
		t.Fatalf("nrem stdout = %q, want two consolidated documents", stdout.String())
	}
	if _, err := os.Stat(memoryPath); err != nil {
		t.Fatalf("configured memory file was not created at %s: %v", memoryPath, err)
	}
	if capturedBaseURL != "http://ollama.example" {
		t.Fatalf("Ollama URL = %q, want config value", capturedBaseURL)
	}
	if capturedModel != "qwen3-embedding" {
		t.Fatalf("embedding model = %q, want config value", capturedModel)
	}
}

func TestNREMUsesXDGConfigFile(t *testing.T) {
	source := t.TempDir()
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nDefault config path should be loaded.\n")
	memoryPath := filepath.Join(t.TempDir(), "xdg-memory.db")
	xdgConfigHome := filepath.Join(t.TempDir(), "xdg")
	home := filepath.Join(t.TempDir(), "home")
	configDir := filepath.Join(xdgConfigHome, "anna")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	writeFile(t, filepath.Join(configDir, "config.toml"), fmt.Sprintf("memory = %q\n", memoryPath))
	t.Setenv("XDG_CONFIG_HOME", xdgConfigHome)
	t.Setenv("HOME", home)

	stdout, stderr, err := executeCommandWithDependencies(defaultTestDependencies(), "nrem", source)
	if err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, memoryPath) {
		t.Fatalf("nrem stdout = %q, want configured memory path %q", stdout, memoryPath)
	}
	if _, err := os.Stat(memoryPath); err != nil {
		t.Fatalf("configured memory file was not created at %s: %v", memoryPath, err)
	}
}

func TestNREMUsesHomeConfigFile(t *testing.T) {
	source := t.TempDir()
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nHome config path should be loaded.\n")
	memoryPath := filepath.Join(t.TempDir(), "home-memory.db")
	home := filepath.Join(t.TempDir(), "home")
	configDir := filepath.Join(home, ".config", "anna")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	writeFile(t, filepath.Join(configDir, "anna.toml"), fmt.Sprintf("memory = %q\n", memoryPath))
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("HOME", home)

	stdout, stderr, err := executeCommandWithDependencies(defaultTestDependencies(), "nrem", source)
	if err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, memoryPath) {
		t.Fatalf("nrem stdout = %q, want configured memory path %q", stdout, memoryPath)
	}
	if _, err := os.Stat(memoryPath); err != nil {
		t.Fatalf("configured memory file was not created at %s: %v", memoryPath, err)
	}
}
