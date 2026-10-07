package cli

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/h3y6e/anna/internal/adapter/fs"
	"github.com/h3y6e/anna/internal/core"
)

func TestNREMWritesTheMemoryInsideTheNotesDirectory(t *testing.T) {
	t.Parallel()

	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nDefault memory path lives next to source files.\n")
	memoryPath := filepath.Join(source, ".anna.db")

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

	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nProgress should be visible during NREM consolidation.\n")
	memoryPath := filepath.Join(source, ".anna.db")

	_, stderr, err := executeCommand("nrem", source)
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
	if !strings.Contains(stderr, "model=Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0") {
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

	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nQuiet mode should suppress progress.\n")

	stdout, stderr, err := executeCommand("nrem", source, "--quiet")
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
	if !strings.Contains(stdout, "--embedder-model") {
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

	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nJSON output should include source and memory paths.\n")

	stdout, stderr, err := executeCommand("nrem", source, "--json")
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

func TestNREMUsesTokenizerFactory(t *testing.T) {
	t.Parallel()

	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nEnglish and 日本語 notes.\n")
	var called bool
	cmd := NewRootCommand("dev", testDependencies(Dependencies{
		TextSource: fs.TextSource{},
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(EmbedderSettings) (core.Embedder, error) {
			return fixedEmbedder{}, nil
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

	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nEnglish and 日本語 notes.\n")
	var capturedModel string
	cmd := NewRootCommand("dev", testDependencies(Dependencies{
		TextSource: fs.TextSource{},
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(s EmbedderSettings) (core.Embedder, error) {
			capturedModel = s.Model
			return fixedEmbedder{}, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return fakeTokenizer{}, nil
		},
	}))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetErr(&stderr)
	cmd.SetArgs([]string{"nrem", source, "--embedder-model", "qwen3-embedding"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr.String())
	}
	if capturedModel != "qwen3-embedding" {
		t.Fatalf("embedding model = %q, want qwen3-embedding", capturedModel)
	}
}

func TestNREMUsesTOMLConfig(t *testing.T) {
	t.Parallel()

	source := tempDir(t)
	writeFile(t, filepath.Join(source, "keep.md"), "# Keep\n\nvisible note\n")
	if err := os.Mkdir(filepath.Join(source, "skip"), 0o700); err != nil {
		t.Fatalf("mkdir skip fixture: %v", err)
	}
	writeFile(t, filepath.Join(source, "skip", "hidden.md"), "# Hidden\n\nthis should now be indexed\n")
	memoryPath := filepath.Join(source, "configured.db")
	configPath := filepath.Join(tempDir(t), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf(`memory = %q
[embedder]
url = "http://embedder.example"
model = "qwen3-embedding"

[nrem]
amnesia = false
`, filepath.Base(memoryPath)))

	var capturedBaseURL string
	var capturedModel string
	cmd := NewRootCommand("dev", testDependencies(Dependencies{
		TextSource: fs.TextSource{},
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(s EmbedderSettings) (core.Embedder, error) {
			capturedBaseURL = s.BaseURL
			capturedModel = s.Model
			return fixedEmbedder{}, nil
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
	if capturedBaseURL != "http://embedder.example" {
		t.Fatalf("embedder URL = %q, want config value", capturedBaseURL)
	}
	if capturedModel != "qwen3-embedding" {
		t.Fatalf("embedding model = %q, want config value", capturedModel)
	}
}

func TestNREMUsesXDGConfigFile(t *testing.T) {
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nDefault config path should be loaded.\n")
	memoryPath := filepath.Join(source, "xdg-memory.db")
	xdgConfigHome := filepath.Join(tempDir(t), "xdg")
	home := filepath.Join(tempDir(t), "home")
	configDir := filepath.Join(xdgConfigHome, "anna")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	writeFile(t, filepath.Join(configDir, "config.toml"), fmt.Sprintf("memory = %q\n", filepath.Base(memoryPath)))
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
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n\nHome config path should be loaded.\n")
	memoryPath := filepath.Join(source, "home-memory.db")
	home := filepath.Join(tempDir(t), "home")
	configDir := filepath.Join(home, ".config", "anna")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	writeFile(t, filepath.Join(configDir, "anna.toml"), fmt.Sprintf("memory = %q\n", filepath.Base(memoryPath)))
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

func TestNREMIndexesEachDirectoryIntoItsOwnMemory(t *testing.T) {
	t.Parallel()

	work := tempDir(t)
	home := tempDir(t)
	writeFile(t, filepath.Join(work, "a.md"), "# A\n")
	writeFile(t, filepath.Join(home, "b.md"), "# B\n")
	writeFile(t, filepath.Join(home, "c.md"), "# C\n")

	stdout, stderr, err := executeCommand("nrem", work, home)
	if err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
	}
	for dir, count := range map[string]int{work: 1, home: 2} {
		index, err := (fs.IndexStore{}).Load(t.Context(), filepath.Join(dir, ".anna.db"))
		if err != nil {
			t.Fatalf("load memory of %s: %v", dir, err)
		}
		if len(index.Documents) != count {
			t.Fatalf("documents in %s = %d, want %d\nstdout: %s", dir, len(index.Documents), count, stdout)
		}
	}
}

func TestNREMDoesNotIndexItsOwnMemoryFile(t *testing.T) {
	t.Parallel()

	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n")

	for range 2 {
		stdout, stderr, err := executeCommand("nrem", source, "--memory", "memory.md")
		if err != nil {
			t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
		}
		if !strings.Contains(stdout, "consolidated 1 documents") {
			t.Fatalf("nrem stdout = %q, want only note.md consolidated", stdout)
		}
	}
}

func TestMemoryMustBeAPlainFileName(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"", ".", "..", "../elsewhere.db", "sub/anna.db", "/abs/memory.db"} {
		_, _, err := executeCommand("nrem", tempDir(t), "--memory", name)
		if err == nil || !strings.Contains(err.Error(), "file name inside each notes directory") {
			t.Fatalf("nrem --memory %q error = %v, want a file-name error", name, err)
		}
	}
}

func TestNREMRejectsNestedDirectoriesBeforeWritingAnyMemory(t *testing.T) {
	t.Parallel()

	vault := tempDir(t)
	task := filepath.Join(vault, "task")
	if err := os.Mkdir(task, 0o700); err != nil {
		t.Fatalf("mkdir task: %v", err)
	}
	writeFile(t, filepath.Join(task, "note.md"), "# Note\n")

	_, _, err := executeCommand("nrem", vault, task)
	if err == nil || !strings.Contains(err.Error(), "overlap") {
		t.Fatalf("nrem error = %v, want overlapping directories error", err)
	}
	for _, dir := range []string{vault, task} {
		if _, err := os.Stat(filepath.Join(dir, ".anna.db")); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("memory in %s exists or stat failed (%v); want nothing written", dir, err)
		}
	}
}

func TestNREMRejectsTheSameDirectoryGivenTwice(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	writeFile(t, filepath.Join(notes, "note.md"), "# Note\n")

	_, _, err := executeCommand("nrem", notes, notes)
	if err == nil || !strings.Contains(err.Error(), "overlap") {
		t.Fatalf("nrem error = %v, want overlapping directories error", err)
	}
}

func TestNREMWithoutArgumentsBuildsTheConfiguredNotes(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	writeFile(t, filepath.Join(notes, "note.md"), "# Note\n")
	configPath := filepath.Join(tempDir(t), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf("notes = [%q]\n", notes))

	_, stderr, err := executeCommand("--config", configPath, "nrem")
	if err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
	}
	if _, err := os.Stat(filepath.Join(notes, ".anna.db")); err != nil {
		t.Fatalf("configured notes memory was not created: %v", err)
	}
}

func TestNREMWithoutArgumentsOrConfiguredNotesExplainsHowToChooseThem(t *testing.T) {
	t.Parallel()

	_, _, err := executeCommand("nrem")
	if err == nil || !strings.Contains(err.Error(), "notes") {
		t.Fatalf("nrem error = %v, want guidance about notes directories", err)
	}
}

func TestNREMIndexesAFileNamedLikeTheMemoryInASubdirectory(t *testing.T) {
	t.Parallel()

	source := tempDir(t)
	writeFile(t, filepath.Join(source, "note.md"), "# Note\n")
	if err := os.Mkdir(filepath.Join(source, "sub"), 0o700); err != nil {
		t.Fatalf("mkdir sub: %v", err)
	}
	writeFile(t, filepath.Join(source, "sub", "memory.md"), "# Sub\n")

	for range 2 {
		stdout, stderr, err := executeCommand("nrem", source, "--memory", "memory.md")
		if err != nil {
			t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
		}
		if !strings.Contains(stdout, "consolidated 2 documents") {
			t.Fatalf("nrem stdout = %q, want note.md and sub/memory.md consolidated", stdout)
		}
	}
}
