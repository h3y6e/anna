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

func TestREMReportsMemoryCandidates(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	memoryPath := filepath.Join(notes, ".anna.db")
	store := fs.IndexStore{}
	if err := store.Save(t.Context(), memoryPath, &core.Index{Version: core.IndexVersion, Documents: []core.Document{
		{
			Path:        "alpha.md",
			Content:     "Retrieval augmented generation keeps local notes searchable.",
			ContentHash: "same",
			Terms:       map[string]int{"retrieval": 1},
			Length:      1,
			Embedding:   []float64{1, 0},
		},
		{
			Path:        "alpha-copy.md",
			Content:     "Retrieval augmented generation keeps local notes searchable.",
			ContentHash: "same",
			Terms:       map[string]int{"retrieval": 1},
			Length:      1,
			Embedding:   []float64{1, 0},
		},
	}}); err != nil {
		t.Fatalf("save fixture memory: %v", err)
	}

	stdout, stderr, err := executeCommand("rem", "--in", notes, "--focus", "echo", "--json")
	if err != nil {
		t.Fatalf("rem command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, `"focus":"echo"`) {
		t.Fatalf("rem stdout = %q, want echo candidate", stdout)
	}
	if !strings.Contains(stdout, fmt.Sprintf(`"left_path":%q`, filepath.Join(notes, "alpha-copy.md"))) {
		t.Fatalf("rem stdout = %q, want stable left path", stdout)
	}
	if !strings.Contains(stdout, fmt.Sprintf(`"right_path":%q`, filepath.Join(notes, "alpha.md"))) {
		t.Fatalf("rem stdout = %q, want stable right path", stdout)
	}
}

func TestREMUsesTOMLConfig(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	memoryPath := filepath.Join(notes, ".anna.db")
	store := fs.IndexStore{}
	if err := store.Save(t.Context(), memoryPath, &core.Index{Version: core.IndexVersion, Documents: []core.Document{
		{
			Path:        "alpha.md",
			ContentHash: "same",
			Embedding:   []float64{1, 0},
		},
		{
			Path:        "alpha-copy.md",
			ContentHash: "same",
			Embedding:   []float64{1, 0},
		},
	}}); err != nil {
		t.Fatalf("save fixture memory: %v", err)
	}
	configPath := filepath.Join(tempDir(t), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf(`notes = [%q]
json = true

[rem]
focus = "echo"
threshold = 1.0
`, notes))

	cmd := NewRootCommand(testDependencies(Dependencies{IndexStore: store}))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetErr(&stderr)
	cmd.SetArgs([]string{"--config", configPath, "rem"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("rem command failed: %v\nstderr: %s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"focus":"echo"`) {
		t.Fatalf("rem stdout = %q, want JSON echo candidate", stdout.String())
	}
}

func TestREMReadsOnlyMemory(t *testing.T) {
	t.Parallel()

	store := &spyIndexStore{index: &core.Index{Documents: []core.Document{
		{
			Path:        "alpha.md",
			ContentHash: "same",
			Embedding:   []float64{1, 0},
		},
		{
			Path:        "alpha-copy.md",
			ContentHash: "same",
			Embedding:   []float64{1, 0},
		},
	}}}
	cmd := NewRootCommand(testDependencies(Dependencies{
		NewTextSource: func() core.TextSource {
			t.Fatal("rem must not read markdown source")
			return nil
		},
		IndexStore: store,
		NewEmbedder: func(EmbedderSettings) (core.Embedder, error) {
			t.Fatal("rem must not create embedder")
			return nil, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			t.Fatal("rem must not create tokenizer")
			return nil, nil
		},
	}))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetErr(&stderr)
	notes := tempDir(t)
	writeFile(t, filepath.Join(notes, ".anna.db"), "")
	cmd.SetArgs([]string{"rem", "--in", notes, "--focus", "echo"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("rem command failed: %v\nstderr: %s", err, stderr.String())
	}
	if want := filepath.Join(notes, ".anna.db"); store.loadedPath != want {
		t.Fatalf("loaded path = %q, want %s", store.loadedPath, want)
	}
	if store.saved {
		t.Fatal("rem saved memory; want read-only behavior")
	}
}

func TestREMRejectsUnsupportedFocus(t *testing.T) {
	t.Parallel()

	notes := tempDir(t)
	saveMemory(t, filepath.Join(notes, ".anna.db"), core.Document{Path: "a.md", Content: "a"})

	_, _, err := executeCommand("rem", "--in", notes, "--focus", "cluster")
	if err == nil || !strings.Contains(err.Error(), `unsupported rem focus "cluster"`) {
		t.Fatalf("rem focus error = %v, want unsupported focus", err)
	}
}

func TestREMWithSeveralConfiguredNotesAsksForOneDirectoryWithIn(t *testing.T) {
	t.Parallel()

	configPath := filepath.Join(tempDir(t), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf("notes = [%q, %q]\n", tempDir(t), tempDir(t)))

	_, _, err := executeCommand("--config", configPath, "rem")
	if err == nil || !strings.Contains(err.Error(), "--in") {
		t.Fatalf("rem error = %v, want guidance to pass --in", err)
	}
}

func TestREMHelpDescribesInAsOneNotesDirectory(t *testing.T) {
	t.Parallel()

	stdout, _, err := executeCommand("rem", "--help")
	if err != nil {
		t.Fatalf("rem help failed: %v", err)
	}
	if !strings.Contains(stdout, "--in string") || strings.Contains(stdout, "repeat") {
		t.Fatalf("rem help = %q, want a single-value --in flag", stdout)
	}
}
