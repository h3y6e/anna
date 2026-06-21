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

	memoryPath := filepath.Join(t.TempDir(), "memory.db")
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

	stdout, stderr, err := executeCommand("rem", "--memory", memoryPath, "--focus", "echo", "--json")
	if err != nil {
		t.Fatalf("rem command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, `"focus":"echo"`) {
		t.Fatalf("rem stdout = %q, want echo candidate", stdout)
	}
	if !strings.Contains(stdout, `"left_path":"alpha-copy.md"`) {
		t.Fatalf("rem stdout = %q, want stable left path", stdout)
	}
	if !strings.Contains(stdout, `"right_path":"alpha.md"`) {
		t.Fatalf("rem stdout = %q, want stable right path", stdout)
	}
}

func TestREMUsesTOMLConfig(t *testing.T) {
	t.Parallel()

	memoryPath := filepath.Join(t.TempDir(), "memory.db")
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
	configPath := filepath.Join(t.TempDir(), "anna.toml")
	writeFile(t, configPath, fmt.Sprintf(`memory = %q
json = true

[rem]
focus = "echo"
threshold = 1.0
`, memoryPath))

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
		NewEmbedder: func(string, string) core.Embedder {
			t.Fatal("rem must not create embedder")
			return nil
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
	cmd.SetArgs([]string{"rem", "--memory", "memory.db", "--focus", "echo"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("rem command failed: %v\nstderr: %s", err, stderr.String())
	}
	if store.loadedPath != "memory.db" {
		t.Fatalf("loaded path = %q, want memory.db", store.loadedPath)
	}
	if store.saved {
		t.Fatal("rem saved memory; want read-only behavior")
	}
}

func TestREMRejectsUnsupportedFocus(t *testing.T) {
	t.Parallel()

	_, _, err := executeCommand("rem", "--memory", "memory.db", "--focus", "cluster")
	if err == nil || !strings.Contains(err.Error(), `unsupported rem focus "cluster"`) {
		t.Fatalf("rem focus error = %v, want unsupported focus", err)
	}
}
