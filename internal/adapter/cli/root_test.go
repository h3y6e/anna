package cli

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/h3y6e/anna/internal/adapter/fs"
	"github.com/h3y6e/anna/internal/core"
)

func TestWhenNotesAreConsolidatedRecallReturnsTheMatchingNoteFirst(t *testing.T) {
	t.Parallel()

	// Arrange
	source := tempDir(t)
	writeFile(t, filepath.Join(source, "ai.md"), "# AI Notes\n\nRetrieval augmented generation keeps local knowledge searchable.\n")
	writeFile(t, filepath.Join(source, "cooking.md"), "# Cooking\n\nMiso soup needs dashi, tofu, and wakame.\n")
	stdout, stderr, err := executeCommand("nrem", source)
	if err != nil {
		t.Fatalf("nrem command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "consolidated 2 documents") {
		t.Fatalf("nrem stdout = %q, want consolidated document count", stdout)
	}

	// Act
	stdout, stderr, err = executeCommand(
		"recall",
		"--in", source,
		"retrieval augmented generation",
		"--limit", "1",
	)

	// Assert
	if err != nil {
		t.Fatalf("recall command failed: %v\nstderr: %s", err, stderr)
	}
	if !strings.Contains(stdout, "ai.md") {
		t.Fatalf("recall stdout = %q, want ai.md", stdout)
	}
	if strings.Contains(stdout, "cooking.md") {
		t.Fatalf("recall stdout = %q, did not want cooking.md in top result", stdout)
	}
}

func executeCommand(args ...string) (string, string, error) {
	return executeCommandWithDependencies(testDependencies(defaultTestDependencies()), args...)
}

func executeCommandWithDependencies(deps Dependencies, args ...string) (string, string, error) {
	cmd := NewRootCommand("dev", deps)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.SetOut(&stdout)
	cmd.SetErr(&stderr)
	cmd.SetArgs(args)
	err := cmd.Execute()
	return stdout.String(), stderr.String(), err
}

func defaultTestDependencies() Dependencies {
	return Dependencies{
		TextSource: fs.TextSource{},
		IndexStore: fs.IndexStore{},
		NewEmbedder: func(EmbedderSettings) (core.Embedder, error) {
			return fakeEmbedder{}, nil
		},
		NewTokenizer: func() (core.Tokenizer, error) {
			return fakeTokenizer{}, nil
		},
	}
}

func testDependencies(deps Dependencies) Dependencies {
	deps.ConfigSearchPaths = []string{}
	return deps
}

func writeFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write fixture %s: %v", path, err)
	}
}

type fixedEmbedder struct{}

func (fixedEmbedder) EmbedQuery(context.Context, string) ([]float64, error) {
	return []float64{1, 0}, nil
}

func (fixedEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i := range out {
		out[i] = []float64{1, 0}
	}
	return out, nil
}

type fakeEmbedder struct{}

func (fakeEmbedder) EmbedQuery(_ context.Context, text string) ([]float64, error) {
	return fakeEmbedderVector(text), nil
}

func (fakeEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i, text := range texts {
		out[i] = fakeEmbedderVector(text)
	}
	return out, nil
}

func fakeEmbedderVector(text string) []float64 {
	text = strings.ToLower(text)
	switch {
	case strings.Contains(text, "retrieval"):
		return []float64{1, 0, 0}
	case strings.Contains(text, "hidden"):
		return []float64{0, 1, 0}
	case strings.Contains(text, "cooking") || strings.Contains(text, "miso"):
		return []float64{0, 0, 1}
	default:
		return []float64{0, 0, 1}
	}
}

type fakeTokenizer struct{}

func (fakeTokenizer) Tokenize(text string) []string {
	return strings.Fields(strings.ToLower(text))
}

// tempDir returns a temporary directory with symlinks resolved, matching the paths anna reports.
func tempDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve temp dir: %v", err)
	}
	return dir
}
