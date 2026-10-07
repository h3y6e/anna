package core

import (
	"path"
	"strings"
)

func countTerms(tokens []string) map[string]int {
	terms := make(map[string]int, len(tokens))
	for _, token := range tokens {
		terms[token]++
	}
	return terms
}

func termCount(terms map[string]int) int {
	var count int
	for _, n := range terms {
		count += n
	}
	return count
}

func documentTitle(doc Document) string {
	if title := extractFrontmatterTitle(doc.Content); title != "" {
		return title
	}
	for line := range strings.SplitSeq(doc.Content, "\n") {
		line = strings.TrimSpace(line)
		if title, ok := strings.CutPrefix(line, "# "); ok {
			return strings.TrimSpace(title)
		}
	}
	return path.Base(strings.TrimSuffix(doc.Path, path.Ext(doc.Path)))
}

func extractFrontmatterTitle(content string) string {
	trimmed := strings.TrimSpace(content)
	if !strings.HasPrefix(trimmed, "---") {
		return ""
	}
	rest := trimmed[len("---"):]
	front, _, ok := strings.Cut(rest, "\n---")
	if !ok {
		return ""
	}
	for line := range strings.SplitSeq(front, "\n") {
		line = strings.TrimSpace(line)
		if title, ok := strings.CutPrefix(line, "title:"); ok {
			title = strings.TrimSpace(title)
			title = strings.Trim(title, "\"'")
			return title
		}
	}
	return ""
}

const maxIndexedTitleRunes = 200

func shortDocumentTitle(doc Document) string {
	title := documentTitle(doc)
	runes := []rune(title)
	if len(runes) > maxIndexedTitleRunes {
		return string(runes[:maxIndexedTitleRunes])
	}
	return title
}
