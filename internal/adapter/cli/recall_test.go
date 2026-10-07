package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/h3y6e/anna/internal/adapter/fs"
	"github.com/h3y6e/anna/internal/core"
)

func TestRecallUsesConfiguredEmbeddingModel(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	saveMemory(t, filepath.Join(notes, ".anna.db"), core.Document{
		Path:      "note.md",
		Terms:     map[string]int{"query": 1},
		Length:    1,
		Embedding: []float64{1, 0},
	})

	var capturedModel string
	_, stderr, err := executeCommandWithDependencies(testDependencies(Dependencies{
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(s EmbedderSettings) (core.Embedder, error) {
			capturedModel = s.Model
			return fixedEmbedder{}, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return fakeTokenizer{}, nil
		},
	}), "recall", "--in", notes, "query", "--embedder-model", "qwen3-embedding")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if capturedModel != "qwen3-embedding" {
		t.Fatalf("embedding model = %q, want qwen3-embedding", capturedModel)
	}
}

func TestRecallSearchesConfiguredNotesDirectories(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	saveMemory(t, filepath.Join(notes, ".anna.db"),
		core.Document{Path: "lexical.md", Content: "exact keyword match", Terms: map[string]int{"keyword": 1}, Length: 1},
		core.Document{Path: "other.md", Content: "different note", Terms: map[string]int{"different": 1}, Length: 1},
	)
	configPath := filepath.Join(tempDir(t), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf(`notes = [%q]
json = true

[recall]
mode = "bm25"
limit = 1
`, notes))

	stdout, stderr, err := executeCommand("--config", configPath, "recall", "keyword")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if want := fmt.Sprintf(`"path":%q`, filepath.Join(notes, "lexical.md")); !strings.Contains(stdout, want) {
		t.Fatalf("recall stdout = %q, want %s", stdout, want)
	}
	if strings.Contains(stdout, "other.md") {
		t.Fatalf("recall stdout = %q, did not want result beyond configured limit", stdout)
	}
}

func TestRecallInFlagOverridesConfiguredNotes(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	saveMemory(t, filepath.Join(notes, ".anna.db"),
		core.Document{Path: "lexical.md", Content: "exact keyword match", Terms: map[string]int{"keyword": 1}, Length: 1},
	)
	configPath := filepath.Join(tempDir(t), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf("notes = [%q]\n\n[recall]\nmode = \"vector\"\n", tempDir(t)))

	stdout, stderr, err := executeCommandWithDependencies(testDependencies(Dependencies{
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(EmbedderSettings) (core.Embedder, error) {
			t.Fatal("flag-selected bm25 mode must not create an embedder")
			return nil, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) { return fakeTokenizer{}, nil },
	}), "--config", configPath, "recall", "keyword", "--in", notes, "--mode", "bm25")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, filepath.Join(notes, "lexical.md")) {
		t.Fatalf("recall stdout = %q, want lexical result", stdout)
	}
}

func TestRecallSearchesSeveralDirectoriesAsOneCorpus(t *testing.T) {
	t.Parallel()

	work := filepath.Join(tempDir(t), "work")
	home := filepath.Join(tempDir(t), "home")
	for _, dir := range []string{work, home} {
		saveMemory(t, filepath.Join(dir, ".anna.db"),
			core.Document{Path: "todo.md", Content: "todo list", Terms: map[string]int{"todo": 1}, Length: 1},
		)
	}

	stdout, stderr, err := executeCommand("recall", "--in", work, "--in", home, "--mode", "bm25", "todo")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	for _, want := range []string{filepath.Join(work, "todo.md"), filepath.Join(home, "todo.md")} {
		if !strings.Contains(stdout, want) {
			t.Fatalf("recall stdout = %q, want %s", stdout, want)
		}
	}
}

func TestRecallRejectsOverlappingDirectories(t *testing.T) {
	t.Parallel()

	vault := tempDir(t)
	task := filepath.Join(vault, "task")
	if err := os.Mkdir(task, 0o700); err != nil {
		t.Fatalf("mkdir task: %v", err)
	}

	_, _, err := executeCommand("recall", "--in", vault, "--in", task, "--mode", "bm25", "query")
	if err == nil || !strings.Contains(err.Error(), "overlap") {
		t.Fatalf("recall error = %v, want overlapping directories error", err)
	}
}

func TestRecallWithoutDirectoriesExplainsHowToChooseThem(t *testing.T) {
	t.Parallel()

	_, _, err := executeCommand("recall", "--mode", "bm25", "query")
	if err == nil || !strings.Contains(err.Error(), "--in") || !strings.Contains(err.Error(), "notes") {
		t.Fatalf("recall error = %v, want guidance about --in and notes", err)
	}
}

func TestRecallReportsAMissingMemoryWithTheNREMCommandToCreateIt(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)

	_, _, err := executeCommand("recall", "--in", notes, "--mode", "bm25", "query")
	if err == nil || !strings.Contains(err.Error(), "anna nrem "+notes) {
		t.Fatalf("recall error = %v, want hint to run anna nrem %s", err, notes)
	}
}

func TestRecallReadsTheMemoryFileNamedByMemory(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	saveMemory(t, filepath.Join(notes, "memory.local.db"),
		core.Document{Path: "note.md", Content: "keyword", Terms: map[string]int{"keyword": 1}, Length: 1},
	)

	stdout, stderr, err := executeCommand("recall", "--memory", "memory.local.db", "--in", notes, "--mode", "bm25", "keyword")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, filepath.Join(notes, "note.md")) {
		t.Fatalf("recall stdout = %q, want note.md", stdout)
	}
}

func TestRecallRejectsUnsupportedMode(t *testing.T) {
	t.Parallel()

	_, _, err := executeCommand("recall", "--in", tempDir(t), "query", "--mode", "unknown")
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
	cwd := tempDir(t)
	t.Chdir(cwd)

	saveMemory(t, filepath.Join(cwd, ".anna.db"),
		core.Document{Path: "note.md", Content: "exact keyword match", Terms: map[string]int{"keyword": 1}, Length: 1},
	)
	writeFile(t, filepath.Join(cwd, "anna.toml"), "notes = [\".\"]\njson = true\n\n[recall]\nmode = \"bm25\"\n")

	stdout, stderr, err := executeCommand("recall", "keyword")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if want := fmt.Sprintf(`"path":%q`, filepath.Join(cwd, "note.md")); !strings.Contains(stdout, want) {
		t.Fatalf("recall stdout = %q, want %s", stdout, want)
	}
}

func saveMemory(t *testing.T, path string, docs ...core.Document) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatalf("mkdir memory dir: %v", err)
	}
	if err := (fs.IndexStore{}).Save(t.Context(), path, &core.Index{Version: core.IndexVersion, Documents: docs}); err != nil {
		t.Fatalf("save fixture memory: %v", err)
	}
}

func TestRecallTreatsASymlinkToANotesDirectoryAsTheSameDirectory(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	link := filepath.Join(tempDir(t), "link")
	if err := os.Symlink(notes, link); err != nil {
		t.Fatalf("symlink: %v", err)
	}

	_, _, err := executeCommand("recall", "--in", notes, "--in", link, "--mode", "bm25", "query")
	if err == nil || !strings.Contains(err.Error(), "overlap") {
		t.Fatalf("recall error = %v, want overlapping directories error", err)
	}
}

func TestRecallSplitsNotesFromTheEnvironmentOnThePathListSeparator(t *testing.T) {
	work := filepath.Join(tempDir(t), "work")
	home := filepath.Join(tempDir(t), "home")
	for _, dir := range []string{work, home} {
		saveMemory(t, filepath.Join(dir, ".anna.db"),
			core.Document{Path: "todo.md", Content: "todo list", Terms: map[string]int{"todo": 1}, Length: 1},
		)
	}
	t.Setenv("ANNA_NOTES", work+string(os.PathListSeparator)+home)

	stdout, stderr, err := executeCommand("recall", "--mode", "bm25", "todo")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	for _, want := range []string{filepath.Join(work, "todo.md"), filepath.Join(home, "todo.md")} {
		if !strings.Contains(stdout, want) {
			t.Fatalf("recall stdout = %q, want %s", stdout, want)
		}
	}
}

func TestRecallResolvesRelativeNotesInAConfigFileAgainstTheConfigFileDirectory(t *testing.T) {
	t.Parallel()

	base := tempDir(t)
	saveMemory(t, filepath.Join(base, "notes", ".anna.db"),
		core.Document{Path: "todo.md", Content: "todo list", Terms: map[string]int{"todo": 1}, Length: 1},
	)
	configPath := filepath.Join(base, "anna.toml")
	writeFile(t, configPath, "notes = [\"notes\"]\n")

	stdout, stderr, err := executeCommand("--config", configPath, "recall", "--mode", "bm25", "todo")
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if want := filepath.Join(base, "notes", "todo.md"); !strings.Contains(stdout, want) {
		t.Fatalf("recall stdout = %q, want %s", stdout, want)
	}
}
