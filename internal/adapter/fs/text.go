package fs

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/h3y6e/anna/internal/core"
	"golang.org/x/sync/errgroup"
)

type TextSource struct{}

func (s TextSource) ReadTextFiles(ctx context.Context, source string) ([]core.TextFile, error) {
	root, err := os.OpenRoot(source)
	if err != nil {
		return nil, fmt.Errorf("open source root: %w", err)
	}
	defer root.Close()

	type entry struct {
		path string
		rel  string
	}
	var entries []entry
	if err := filepath.WalkDir(source, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if d.IsDir() || !d.Type().IsRegular() {
			return nil
		}
		if !isTextDocument(path) {
			return nil
		}
		rel, err := filepath.Rel(source, path)
		if err != nil {
			return fmt.Errorf("relativize %s: %w", path, err)
		}
		entries = append(entries, entry{path: path, rel: rel})
		return nil
	}); err != nil {
		return nil, fmt.Errorf("walk source: %w", err)
	}

	files := make([]core.TextFile, len(entries))
	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(32)
	for i, e := range entries {
		g.Go(func() error {
			if err := ctx.Err(); err != nil {
				return err
			}
			content, err := root.ReadFile(e.rel)
			if err != nil {
				return fmt.Errorf("read %s: %w", e.path, err)
			}
			files[i] = core.TextFile{Path: filepath.ToSlash(e.rel), Content: string(content)}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}

	slices.SortFunc(files, func(a, b core.TextFile) int { return cmp.Compare(a.Path, b.Path) })
	return files, nil
}

func isTextDocument(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".md", ".markdown", ".mdx", ".rst", ".txt", ".text":
		return true
	default:
		return false
	}
}
