package core

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"
)

type Searcher struct {
	store     IndexStore
	embedder  Embedder
	tokenizer Tokenizer

	embedding EmbeddingProfile
}

type SearchMode string

const (
	SearchModeBM25   SearchMode = "bm25"
	SearchModeVector SearchMode = "vector"
	SearchModeHybrid SearchMode = "hybrid"
	SearchModeRRF    SearchMode = "rrf"
)

const rrfK = 60.0

func ParseSearchMode(value string) (SearchMode, error) {
	mode := SearchMode(strings.TrimSpace(value))
	if err := mode.Validate(); err != nil {
		return "", err
	}
	return mode, nil
}

func (m SearchMode) Validate() error {
	switch m {
	case SearchModeBM25, SearchModeVector, SearchModeHybrid, SearchModeRRF:
		return nil
	default:
		return fmt.Errorf("unsupported search mode %q", m)
	}
}

func (m SearchMode) RequiresEmbedding() bool {
	return m == SearchModeVector || m == SearchModeHybrid || m == SearchModeRRF
}

func NewSearcher(store IndexStore, embedder Embedder, tokenizer Tokenizer) *Searcher {
	return &Searcher{store: store, embedder: embedder, tokenizer: tokenizer}
}

func (s *Searcher) WithEmbedding(embedding EmbeddingProfile) *Searcher {
	s.embedding = embedding
	return s
}

// SearchFiles searches several memory files as one corpus. Document paths in the results are joined to the
// directory that holds each memory file.
func (s *Searcher) SearchFiles(
	ctx context.Context,
	indexPaths []string,
	query string,
	limit int,
	mode SearchMode,
) ([]SearchResult, error) {
	if err := mode.Validate(); err != nil {
		return nil, err
	}
	queryTerms := unique(s.tokenizer.Tokenize(query))

	var docs []Document
	for _, indexPath := range indexPaths {
		embedding, memoryDocs, err := s.store.LoadSearchDocuments(ctx, indexPath, queryTerms, mode.RequiresEmbedding())
		if err != nil {
			return nil, fmt.Errorf("%s: %w", indexPath, err)
		}
		if mode.RequiresEmbedding() {
			if err := validateSearchEmbedding(embedding, s.embedding); err != nil {
				return nil, fmt.Errorf("%s: %w", indexPath, err)
			}
		}
		for _, doc := range memoryDocs {
			doc.Path = resolveDocumentPath(indexPath, doc.Path)
			docs = append(docs, doc)
		}
	}
	slices.SortFunc(docs, func(a, b Document) int { return cmp.Compare(a.Path, b.Path) })

	var queryEmbedding []float64
	if mode.RequiresEmbedding() {
		var err error
		queryEmbedding, err = s.embedder.EmbedQuery(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embed query: %w", err)
		}
		if err := validateSearchEmbeddings(docs, queryEmbedding); err != nil {
			return nil, err
		}
	}
	return searchTokenized(docs, query, queryTerms, queryEmbedding, limit, mode), nil
}

// resolveDocumentPath joins a document path stored in a memory file to the directory that holds the memory file.
func resolveDocumentPath(indexPath string, documentPath string) string {
	return filepath.Join(filepath.Dir(indexPath), filepath.FromSlash(documentPath))
}

func validateSearchEmbedding(recorded EmbeddingProfile, configured EmbeddingProfile) error {
	if recorded == configured {
		return nil
	}
	return fmt.Errorf(
		"index was built with --embedder-model %q --embedder-query-prefix %q --embedder-document-prefix %q; search with the same settings or rebuild the index",
		recorded.Model,
		recorded.QueryPrefix,
		recorded.DocumentPrefix,
	)
}

func validateSearchEmbeddings(docs []Document, queryEmbedding []float64) error {
	if len(queryEmbedding) == 0 {
		return fmt.Errorf("query embedding is empty")
	}
	for _, doc := range docs {
		if len(doc.Embedding) == 0 {
			return fmt.Errorf("index document %s has no embedding; rebuild index", doc.Path)
		}
		if len(doc.Embedding) != len(queryEmbedding) {
			return fmt.Errorf(
				"index document %s embedding dimensions %d do not match query dimensions %d; rebuild index",
				doc.Path,
				len(doc.Embedding),
				len(queryEmbedding),
			)
		}
	}
	return nil
}

func searchTokenized(
	docs []Document,
	query string,
	queryTerms []string,
	queryEmbedding []float64,
	limit int,
	mode SearchMode,
) []SearchResult {
	if mode == SearchModeRRF {
		return searchRRF(docs, query, queryTerms, queryEmbedding, limit)
	}

	df := documentFrequencies(docs, queryTerms)
	docCount := len(docs)
	avgLen := averageLength(docs)
	lowerQuery := strings.ToLower(strings.TrimSpace(query))
	results := make([]SearchResult, 0, len(docs))
	for _, doc := range docs {
		bm25Score := bm25(docCount, doc, queryTerms, df, avgLen)
		lexicalScore := bm25Score + phraseBoost(doc, lowerQuery)
		var score float64
		vectorScore, hasVector := cosineSimilarity(queryEmbedding, doc.Embedding)
		semanticScore := max(0, vectorScore)
		switch mode {
		case SearchModeBM25:
			score = lexicalScore
		case SearchModeVector:
			if hasVector {
				score = semanticScore
			}
		case SearchModeHybrid:
			if hasVector {
				score = 0.8*semanticScore + 0.2*(bm25Score/(bm25Score+1))
			} else {
				score = lexicalScore
			}
		}
		if score <= 0 {
			continue
		}
		results = append(results, SearchResult{
			Path:    doc.Path,
			Score:   score,
			Snippet: snippet(doc.Content, query),
		})
	}

	sortSearchResults(results)
	if len(results) > limit {
		results = results[:limit]
	}
	return results
}

type rrfScoredDocument struct {
	doc      Document
	bm25     float64
	semantic float64
}

type rrfFusedDocument struct {
	doc      Document
	rrf      float64
	semantic float64
}

func searchRRF(docs []Document, query string, queryTerms []string, queryEmbedding []float64, limit int) []SearchResult {
	df := documentFrequencies(docs, queryTerms)
	docCount := len(docs)
	avgLen := averageLength(docs)
	keywordList := make([]rrfScoredDocument, 0, len(docs))
	vectorList := make([]rrfScoredDocument, 0, len(docs))
	for _, doc := range docs {
		item := rrfScoredDocument{
			doc:  doc,
			bm25: bm25(docCount, doc, queryTerms, df, avgLen),
		}
		if vectorScore, ok := cosineSimilarity(queryEmbedding, doc.Embedding); ok {
			item.semantic = max(0, vectorScore)
		}
		if item.bm25 > 0 {
			keywordList = append(keywordList, item)
		}
		if item.semantic > 0 {
			vectorList = append(vectorList, item)
		}
	}
	sortRRFList(keywordList, func(doc rrfScoredDocument) float64 { return doc.bm25 })
	sortRRFList(vectorList, func(doc rrfScoredDocument) float64 { return doc.semantic })

	fused := make(map[string]rrfFusedDocument, len(docs))
	addRRF(fused, keywordList)
	addRRF(fused, vectorList)
	if len(fused) == 0 {
		return nil
	}

	var maxRRF float64
	for _, item := range fused {
		maxRRF = max(maxRRF, item.rrf)
	}
	results := make([]SearchResult, 0, len(fused))
	for _, item := range fused {
		normalizedRRF := 0.0
		if maxRRF > 0 {
			normalizedRRF = item.rrf / maxRRF
		}
		score := 0.7*normalizedRRF + 0.3*item.semantic
		if score <= 0 {
			continue
		}
		results = append(results, SearchResult{
			Path:    item.doc.Path,
			Score:   score,
			Snippet: snippet(item.doc.Content, query),
		})
	}
	sortSearchResults(results)
	if len(results) > limit {
		results = results[:limit]
	}
	return results
}

func sortRRFList(items []rrfScoredDocument, score func(rrfScoredDocument) float64) {
	slices.SortFunc(items, func(a, b rrfScoredDocument) int {
		if c := cmp.Compare(score(b), score(a)); c != 0 {
			return c
		}
		return cmp.Compare(a.doc.Path, b.doc.Path)
	})
}

func addRRF(fused map[string]rrfFusedDocument, list []rrfScoredDocument) {
	for rank, item := range list {
		entry := fused[item.doc.Path]
		if entry.doc.Path == "" {
			entry.doc = item.doc
		}
		entry.rrf += 1 / (rrfK + float64(rank))
		entry.semantic = item.semantic
		fused[item.doc.Path] = entry
	}
}

func sortSearchResults(results []SearchResult) {
	slices.SortFunc(results, func(a, b SearchResult) int {
		if c := cmp.Compare(b.Score, a.Score); c != 0 {
			return c
		}
		return cmp.Compare(a.Path, b.Path)
	})
}

func cosineSimilarity(a []float64, b []float64) (float64, bool) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, false
	}
	var dot float64
	var aNorm float64
	var bNorm float64
	for i, av := range a {
		bv := b[i]
		dot += av * bv
		aNorm += av * av
		bNorm += bv * bv
	}
	denom := aNorm * bNorm
	if denom == 0 {
		return 0, false
	}
	return dot / math.Sqrt(denom), true
}

func documentFrequencies(docs []Document, terms []string) map[string]int {
	df := make(map[string]int, len(terms))
	for _, term := range terms {
		for _, doc := range docs {
			if doc.Terms[term] > 0 {
				df[term]++
			}
		}
	}
	return df
}

func bm25(docCount int, doc Document, terms []string, df map[string]int, avgLen float64) float64 {
	const k1 = 1.5
	const b = 0.75

	var score float64
	for _, term := range terms {
		tf := float64(doc.Terms[term])
		if tf == 0 {
			continue
		}
		idf := math.Log(1 + (float64(docCount-df[term])+0.5)/(float64(df[term])+0.5))
		denom := tf + k1*(1-b+b*float64(doc.Length)/avgLen)
		score += idf * (tf * (k1 + 1)) / denom
	}
	return score
}

func phraseBoost(doc Document, lowerQuery string) float64 {
	if lowerQuery == "" {
		return 0
	}
	if strings.Contains(strings.ToLower(documentTitle(doc)), lowerQuery) {
		return 3
	}
	if strings.Contains(strings.ToLower(doc.Content), lowerQuery) {
		return 1.5
	}
	return 0
}

func averageLength(docs []Document) float64 {
	var total int
	for _, doc := range docs {
		total += doc.Length
	}
	if total == 0 {
		return 1
	}
	return float64(total) / float64(len(docs))
}

func unique(tokens []string) []string {
	seen := make(map[string]bool, len(tokens))
	terms := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if seen[token] {
			continue
		}
		seen[token] = true
		terms = append(terms, token)
	}
	return terms
}

func snippet(content string, query string) string {
	cleaned := strings.Join(strings.Fields(content), " ")
	if len([]rune(cleaned)) <= 180 {
		return cleaned
	}

	index := strings.Index(strings.ToLower(cleaned), strings.ToLower(strings.TrimSpace(query)))
	if index < 0 {
		return string([]rune(cleaned)[:180])
	}

	runes := []rune(cleaned)
	start := max(0, runeIndex(cleaned, index)-60)
	end := min(len(runes), start+180)
	return strings.TrimSpace(string(runes[start:end]))
}

func runeIndex(s string, byteIndex int) int {
	return len([]rune(s[:byteIndex]))
}
