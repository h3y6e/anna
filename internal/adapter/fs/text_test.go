package fs

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTextSourceSkipsDirectoriesAndSymlinks(t *testing.T) {
	t.Parallel()

	source := t.TempDir()
	if err := os.Mkdir(filepath.Join(source, "dir"), 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.Symlink(filepath.Join(source, "dir"), filepath.Join(source, "link")); err != nil {
		t.Fatalf("symlink: %v", err)
	}
	writeTextFile(t, filepath.Join(source, "note.md"), "note")

	files, err := TextSource{}.ReadTextFiles(t.Context(), source)
	if err != nil {
		t.Fatalf("ReadTextFiles error: %v", err)
	}
	if len(files) != 1 || files[0].Path != "note.md" {
		t.Fatalf("files = %+v, want only note.md", files)
	}
}

func TestTextSourceIncludesAllowedExtensions(t *testing.T) {
	t.Parallel()

	source := t.TempDir()
	writeTextFile(t, filepath.Join(source, "note.md"), "# Note\n\nmarkdown note")
	writeTextFile(t, filepath.Join(source, "memo.txt"), "plain text memo")
	writeTextFile(t, filepath.Join(source, "component.mdx"), "# Component\n\nJSX mixed")
	writeTextFile(t, filepath.Join(source, "guide.rst"), "Guide\n=====\n\nrst content")
	writeTextFile(t, filepath.Join(source, "empty.md"), "")
	writeTextFile(t, filepath.Join(source, "data.json"), "{}")
	writeTextFile(t, filepath.Join(source, "page.html"), "<p>html</p>")
	writeTextFile(t, filepath.Join(source, "noext"), "no extension text")
	writeBinaryFile(t, filepath.Join(source, "image.bin"))

	files, err := TextSource{}.ReadTextFiles(t.Context(), source)
	if err != nil {
		t.Fatalf("ReadTextFiles error: %v", err)
	}
	paths := make(map[string]bool, len(files))
	for _, f := range files {
		paths[f.Path] = true
	}

	for _, want := range []string{"note.md", "memo.txt", "component.mdx", "guide.rst", "empty.md"} {
		if !paths[want] {
			t.Fatalf("missing %s", want)
		}
	}
	for _, reject := range []string{"data.json", "page.html", "noext", "image.bin"} {
		if paths[reject] {
			t.Fatalf("%s should not be indexed", reject)
		}
	}
}

func writeTextFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write fixture %s: %v", path, err)
	}
}

func writeBinaryFile(t *testing.T, path string) {
	t.Helper()
	if err := os.WriteFile(path, []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00}, 0o600); err != nil {
		t.Fatalf("write fixture %s: %v", path, err)
	}
}
