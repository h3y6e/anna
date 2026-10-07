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

func TestWhenTheEmbedderModelFlagIsGivenRecallPassesItToTheEmbedder(t *testing.T) {
	t.Parallel()

	// Arrange
	notes := tempDir(t)
	saveMemory(t, filepath.Join(notes, ".anna.db"), "qwen3-embedding", core.Document{
		Path:      "note.md",
		Terms:     map[string]int{"query": 1},
		Length:    1,
		Embedding: []float64{1, 0},
	})

	var capturedModel string

	// Act
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

	// Assert
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if capturedModel != "qwen3-embedding" {
		t.Fatalf("embedding model = %q, want qwen3-embedding", capturedModel)
	}
}

func TestWhenTheConfigListsNotesRecallSearchesThemWithTheConfiguredModeAndLimit(t *testing.T) {
	t.Parallel()

	// Arrange
	notes := tempDir(t)
	saveMemory(t, filepath.Join(notes, ".anna.db"), "",
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

	// Act
	stdout, stderr, err := executeCommand("--config", configPath, "recall", "keyword")

	// Assert
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

func TestWhenInAndModeFlagsAreGivenRecallUsesThemOverTheConfig(t *testing.T) {
	t.Parallel()

	// Arrange
	notes := tempDir(t)
	saveMemory(t, filepath.Join(notes, ".anna.db"), "",
		core.Document{Path: "lexical.md", Content: "exact keyword match", Terms: map[string]int{"keyword": 1}, Length: 1},
	)
	configPath := filepath.Join(tempDir(t), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf("notes = [%q]\n\n[recall]\nmode = \"vector\"\n", tempDir(t)))

	// Act
	stdout, stderr, err := executeCommandWithDependencies(testDependencies(Dependencies{
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(EmbedderSettings) (core.Embedder, error) {
			t.Fatal("flag-selected bm25 mode must not create an embedder")
			return nil, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) { return fakeTokenizer{}, nil },
	}), "--config", configPath, "recall", "keyword", "--in", notes, "--mode", "bm25")

	// Assert
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, filepath.Join(notes, "lexical.md")) {
		t.Fatalf("recall stdout = %q, want lexical result", stdout)
	}
}

func TestWhenSeveralInFlagsAreGivenRecallReturnsNotesFromEveryDirectory(t *testing.T) {
	t.Parallel()

	// Arrange
	work := filepath.Join(tempDir(t), "work")
	home := filepath.Join(tempDir(t), "home")
	for _, dir := range []string{work, home} {
		saveMemory(t, filepath.Join(dir, ".anna.db"), "",
			core.Document{Path: "todo.md", Content: "todo list", Terms: map[string]int{"todo": 1}, Length: 1},
		)
	}

	// Act
	stdout, stderr, err := executeCommand("recall", "--in", work, "--in", home, "--mode", "bm25", "todo")

	// Assert
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	for _, want := range []string{filepath.Join(work, "todo.md"), filepath.Join(home, "todo.md")} {
		if !strings.Contains(stdout, want) {
			t.Fatalf("recall stdout = %q, want %s", stdout, want)
		}
	}
}

func TestWhenTheDirectoriesOverlapRecallFailsWithAnOverlapError(t *testing.T) {
	t.Parallel()

	// Arrange
	vault := tempDir(t)
	task := filepath.Join(vault, "task")
	if err := os.Mkdir(task, 0o700); err != nil {
		t.Fatalf("mkdir task: %v", err)
	}

	// Act
	_, _, err := executeCommand("recall", "--in", vault, "--in", task, "--mode", "bm25", "query")

	// Assert
	if err == nil || !strings.Contains(err.Error(), "overlap") {
		t.Fatalf("recall error = %v, want overlapping directories error", err)
	}
}

func TestWhenNoDirectoryIsGivenOrConfiguredRecallFailsWithGuidanceAboutInAndNotes(t *testing.T) {
	t.Parallel()

	// Act
	_, _, err := executeCommand("recall", "--mode", "bm25", "query")

	// Assert
	if err == nil || !strings.Contains(err.Error(), "--in") || !strings.Contains(err.Error(), "notes") {
		t.Fatalf("recall error = %v, want guidance about --in and notes", err)
	}
}

func TestWhenTheMemoryIsMissingRecallFailsWithTheNremHint(t *testing.T) {
	t.Parallel()

	// Arrange
	notes := tempDir(t)

	// Act
	_, _, err := executeCommand("recall", "--in", notes, "--mode", "bm25", "query")

	// Assert
	if err == nil || !strings.Contains(err.Error(), "anna nrem "+notes) {
		t.Fatalf("recall error = %v, want hint to run anna nrem %s", err, notes)
	}
}

func TestWhenTheMemoryFlagNamesAFileRecallReadsThatFile(t *testing.T) {
	t.Parallel()

	// Arrange
	notes := tempDir(t)
	saveMemory(t, filepath.Join(notes, "memory.local.db"), "",
		core.Document{Path: "note.md", Content: "keyword", Terms: map[string]int{"keyword": 1}, Length: 1},
	)

	// Act
	stdout, stderr, err := executeCommand("recall", "--memory", "memory.local.db", "--in", notes, "--mode", "bm25", "keyword")

	// Assert
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, filepath.Join(notes, "note.md")) {
		t.Fatalf("recall stdout = %q, want note.md", stdout)
	}
}

func TestWhenTheModeIsUnsupportedRecallFailsWithTheModeName(t *testing.T) {
	t.Parallel()

	// Act
	_, _, err := executeCommand("recall", "--in", tempDir(t), "query", "--mode", "unknown")

	// Assert
	if err == nil || !strings.Contains(err.Error(), `unsupported search mode "unknown"`) {
		t.Fatalf("recall mode error = %v, want unsupported mode", err)
	}
}

func TestWhenHelpIsRequestedRecallListsTheModeFlagAndItsChoices(t *testing.T) {
	t.Parallel()

	// Act
	stdout, stderr, err := executeCommand("recall", "--help")

	// Assert
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

func TestWhenTheWorkingDirectoryHasAConfigRecallUsesIt(t *testing.T) {
	// Arrange
	cwd := tempDir(t)
	t.Chdir(cwd)

	saveMemory(t, filepath.Join(cwd, ".anna.db"), "",
		core.Document{Path: "note.md", Content: "exact keyword match", Terms: map[string]int{"keyword": 1}, Length: 1},
	)
	writeFile(t, filepath.Join(cwd, "anna.toml"), "notes = [\".\"]\njson = true\n\n[recall]\nmode = \"bm25\"\n")

	// Act
	stdout, stderr, err := executeCommand("recall", "keyword")

	// Assert
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if want := fmt.Sprintf(`"path":%q`, filepath.Join(cwd, "note.md")); !strings.Contains(stdout, want) {
		t.Fatalf("recall stdout = %q, want %s", stdout, want)
	}
}

func saveMemory(t *testing.T, path string, model string, docs ...core.Document) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatalf("mkdir memory dir: %v", err)
	}
	if err := (fs.IndexStore{}).Save(t.Context(), path, &core.Index{Embedding: core.EmbeddingProfile{Model: model}, Documents: docs}); err != nil {
		t.Fatalf("save fixture memory: %v", err)
	}
}

func TestWhenASymlinkPointsToAnotherGivenDirectoryRecallFailsWithAnOverlapError(t *testing.T) {
	t.Parallel()

	// Arrange
	notes := tempDir(t)
	link := filepath.Join(tempDir(t), "link")
	if err := os.Symlink(notes, link); err != nil {
		t.Fatalf("symlink: %v", err)
	}

	// Act
	_, _, err := executeCommand("recall", "--in", notes, "--in", link, "--mode", "bm25", "query")

	// Assert
	if err == nil || !strings.Contains(err.Error(), "overlap") {
		t.Fatalf("recall error = %v, want overlapping directories error", err)
	}
}

func TestWhenTheNotesEnvironmentVariableListsSeveralDirectoriesRecallSearchesEach(t *testing.T) {
	// Arrange
	work := filepath.Join(tempDir(t), "work")
	home := filepath.Join(tempDir(t), "home")
	for _, dir := range []string{work, home} {
		saveMemory(t, filepath.Join(dir, ".anna.db"), "",
			core.Document{Path: "todo.md", Content: "todo list", Terms: map[string]int{"todo": 1}, Length: 1},
		)
	}
	t.Setenv("ANNA_NOTES", work+string(os.PathListSeparator)+home)

	// Act
	stdout, stderr, err := executeCommand("recall", "--mode", "bm25", "todo")

	// Assert
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	for _, want := range []string{filepath.Join(work, "todo.md"), filepath.Join(home, "todo.md")} {
		if !strings.Contains(stdout, want) {
			t.Fatalf("recall stdout = %q, want %s", stdout, want)
		}
	}
}

func TestWhenTheConfigListsARelativeNotesDirectoryRecallResolvesItAgainstTheConfigDirectory(t *testing.T) {
	t.Parallel()

	// Arrange
	base := tempDir(t)
	saveMemory(t, filepath.Join(base, "notes", ".anna.db"), "",
		core.Document{Path: "todo.md", Content: "todo list", Terms: map[string]int{"todo": 1}, Length: 1},
	)
	configPath := filepath.Join(base, "anna.toml")
	writeFile(t, configPath, "notes = [\"notes\"]\n")

	// Act
	stdout, stderr, err := executeCommand("--config", configPath, "recall", "--mode", "bm25", "todo")

	// Assert
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if want := filepath.Join(base, "notes", "todo.md"); !strings.Contains(stdout, want) {
		t.Fatalf("recall stdout = %q, want %s", stdout, want)
	}
}
