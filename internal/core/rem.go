package core

import (
	"cmp"
	"context"
	"fmt"
	"path"
	"regexp"
	"slices"
	"strings"
)

const remEchoSimilarityThreshold = 0.95

var remWikiLinkPattern = regexp.MustCompile(`\[\[([^\]]+)\]\]`)

type REMFocus string

const (
	REMFocusAll     REMFocus = "all"
	REMFocusEcho    REMFocus = "echo"
	REMFocusSynapse REMFocus = "synapse"
)

type REMOptions struct {
	Focus     REMFocus
	Limit     int
	Threshold float64
}

type REMCandidate struct {
	Focus     REMFocus `json:"focus"`
	LeftPath  string   `json:"left_path"`
	RightPath string   `json:"right_path"`
	Score     float64  `json:"score"`
	Reason    string   `json:"reason"`
}

type REMer struct {
	store IndexStore
}

func NewREMer(store IndexStore) *REMer {
	return &REMer{store: store}
}

func (r *REMer) REM(ctx context.Context, memoryPath string, options REMOptions) ([]REMCandidate, error) {
	if memoryPath == "" {
		return nil, fmt.Errorf("memory path is required")
	}
	if r.store == nil {
		return nil, fmt.Errorf("memory store is required")
	}
	options, err := normalizeREMOptions(options)
	if err != nil {
		return nil, err
	}
	memory, err := r.store.Load(ctx, memoryPath)
	if err != nil {
		return nil, err
	}
	return REM(memory, options)
}

func REM(memory *Index, options REMOptions) ([]REMCandidate, error) {
	options, err := normalizeREMOptions(options)
	if err != nil {
		return nil, err
	}
	if memory == nil || len(memory.Documents) == 0 {
		return nil, nil
	}

	pairs := remPairs(memory.Documents, options.Threshold)
	var candidates []REMCandidate
	if options.Focus == REMFocusEcho || options.Focus == REMFocusAll {
		candidates = append(candidates, remCandidatesForFocus(pairs, REMFocusEcho)...)
	}
	if options.Focus == REMFocusSynapse || options.Focus == REMFocusAll {
		candidates = append(candidates, remCandidatesForFocus(pairs, REMFocusSynapse)...)
	}
	if len(candidates) > options.Limit {
		candidates = candidates[:options.Limit]
	}
	return candidates, nil
}

type remPair struct {
	left        Document
	right       Document
	score       float64
	sameContent bool
}

func normalizeREMOptions(options REMOptions) (REMOptions, error) {
	if options.Focus == "" {
		options.Focus = REMFocusAll
	}
	switch options.Focus {
	case REMFocusAll, REMFocusEcho, REMFocusSynapse:
	default:
		return REMOptions{}, fmt.Errorf("unsupported rem focus %q", options.Focus)
	}
	if options.Limit <= 0 {
		options.Limit = 10
	}
	if options.Threshold < 0 || options.Threshold > 1 {
		return REMOptions{}, fmt.Errorf("threshold must be between 0 and 1")
	}
	return options, nil
}

func remPairs(documents []Document, threshold float64) []remPair {
	docs := append([]Document(nil), documents...)
	slices.SortFunc(docs, func(a, b Document) int { return cmp.Compare(a.Path, b.Path) })

	pairs := make([]remPair, 0, len(docs))
	for i, left := range docs {
		for _, right := range docs[i+1:] {
			sameContent := left.ContentHash != "" && left.ContentHash == right.ContentHash
			score, ok := cosineSimilarity(left.Embedding, right.Embedding)
			needsPerfectScore := sameContent && (!ok || score < 1)
			if needsPerfectScore {
				score = 1
				ok = true
			}
			if !ok || score < threshold {
				continue
			}
			pairs = append(pairs, remPair{left: left, right: right, score: score, sameContent: sameContent})
		}
	}
	return pairs
}

func remCandidatesForFocus(pairs []remPair, focus REMFocus) []REMCandidate {
	candidates := make([]REMCandidate, 0, len(pairs))
	for _, pair := range pairs {
		switch focus {
		case REMFocusEcho:
			if !pair.sameContent && pair.score < remEchoSimilarityThreshold {
				continue
			}
			candidates = append(candidates, REMCandidate{
				Focus:     REMFocusEcho,
				LeftPath:  pair.left.Path,
				RightPath: pair.right.Path,
				Score:     pair.score,
				Reason:    remEchoReason(pair),
			})
		case REMFocusSynapse:
			isEcho := pair.sameContent || pair.score >= remEchoSimilarityThreshold
			if isEcho || documentsExplicitlyLinked(pair.left, pair.right) {
				continue
			}
			candidates = append(candidates, REMCandidate{
				Focus:     REMFocusSynapse,
				LeftPath:  pair.left.Path,
				RightPath: pair.right.Path,
				Score:     pair.score,
				Reason:    "similar embeddings without an explicit wikilink",
			})
		}
	}
	sortREMCandidates(candidates)
	return candidates
}

func remEchoReason(pair remPair) string {
	if pair.sameContent {
		return "matching content hash"
	}
	return "duplicate-strength embedding similarity"
}

func documentsExplicitlyLinked(left Document, right Document) bool {
	return documentLinksTo(left, right) || documentLinksTo(right, left)
}

func documentLinksTo(source Document, target Document) bool {
	keys := documentLinkKeys(target)
	for _, match := range remWikiLinkPattern.FindAllStringSubmatch(source.Content, -1) {
		if keys[normalizeLinkKey(cleanWikiTarget(match[1]))] {
			return true
		}
	}
	return false
}

func documentLinkKeys(doc Document) map[string]bool {
	withoutExt := strings.TrimSuffix(doc.Path, path.Ext(doc.Path))
	return map[string]bool{
		normalizeLinkKey(withoutExt):            true,
		normalizeLinkKey(path.Base(withoutExt)): true,
		normalizeLinkKey(documentTitle(doc)):    true,
	}
}

func documentTitle(doc Document) string {
	title, _ := documentTitleWithSource(doc)
	return title
}

func documentTitleWithSource(doc Document) (string, bool) {
	if title := extractFrontmatterTitle(doc.Content); title != "" {
		return title, true
	}
	for line := range strings.SplitSeq(doc.Content, "\n") {
		line = strings.TrimSpace(line)
		if title, ok := strings.CutPrefix(line, "# "); ok {
			return strings.TrimSpace(title), false
		}
	}
	return path.Base(strings.TrimSuffix(doc.Path, path.Ext(doc.Path))), false
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

func cleanWikiTarget(target string) string {
	target, _, _ = strings.Cut(target, "|")
	target = strings.TrimSpace(target)
	target, _, _ = strings.Cut(target, "#")
	target = strings.TrimSpace(target)
	target = strings.TrimSuffix(target, ".md")
	return strings.Trim(target, "/")
}

func normalizeLinkKey(target string) string {
	target = path.Clean(strings.TrimSpace(strings.ToLower(target)))
	if target == "." {
		return ""
	}
	target = strings.TrimSuffix(target, ".md")
	target = strings.Trim(target, "/")
	return target
}

func sortREMCandidates(candidates []REMCandidate) {
	slices.SortFunc(candidates, func(a, b REMCandidate) int {
		if c := cmp.Compare(b.Score, a.Score); c != 0 {
			return c
		}
		if c := cmp.Compare(a.LeftPath, b.LeftPath); c != 0 {
			return c
		}
		if c := cmp.Compare(a.RightPath, b.RightPath); c != 0 {
			return c
		}
		return cmp.Compare(a.Focus, b.Focus)
	})
}
