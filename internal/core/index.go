package core

import (
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	"golang.org/x/sync/errgroup"
)

type IndexProgress struct {
	Current int
	Total   int
	Path    string
	Cached  bool
}

type Indexer struct {
	source    TextSource
	store     IndexStore
	embedder  Embedder
	tokenizer Tokenizer

	embedding   EmbeddingProfile
	ignoredPath string
	progress    func(IndexProgress)
}

func NewIndexer(source TextSource, store IndexStore, embedder Embedder, tokenizer Tokenizer) *Indexer {
	return &Indexer{source: source, store: store, embedder: embedder, tokenizer: tokenizer}
}

func (i *Indexer) WithEmbedding(embedding EmbeddingProfile) *Indexer {
	embedding.Model = strings.TrimSpace(embedding.Model)
	i.embedding = embedding
	return i
}

// WithIgnoredPath skips the file at this path relative to the source, such as the memory stored in the notes directory.
func (i *Indexer) WithIgnoredPath(relative string) *Indexer {
	i.ignoredPath = relative
	return i
}

func (i *Indexer) WithProgress(fn func(IndexProgress)) *Indexer {
	var mu sync.Mutex
	i.progress = func(p IndexProgress) {
		mu.Lock()
		defer mu.Unlock()
		fn(p)
	}
	return i
}

func (i *Indexer) BuildAndSave(ctx context.Context, source string, indexPath string, rebuild bool) (*Index, error) {
	if !rebuild {
		return i.buildIncremental(ctx, source, indexPath)
	}
	index, err := i.build(ctx, source)
	if err != nil {
		return nil, err
	}
	if err := i.store.Save(ctx, indexPath, index); err != nil {
		return nil, err
	}
	return index, nil
}

func (i *Indexer) build(ctx context.Context, source string) (*Index, error) {
	files, err := i.readTextFiles(ctx, source)
	if err != nil {
		return nil, err
	}
	docs, err := i.buildDocuments(ctx, files, nil, contentHashes(files))
	if err != nil {
		return nil, err
	}
	return i.newIndex(docs), nil
}

func (i *Indexer) buildIncremental(ctx context.Context, source string, indexPath string) (*Index, error) {
	files, err := i.readTextFiles(ctx, source)
	if err != nil {
		return nil, err
	}
	hashes := contentHashes(files)
	manifest, err := i.store.LoadManifest(ctx, indexPath)
	if err == nil && i.canReuseManifest(manifest) && manifestMatchesFiles(manifest, files, hashes) {
		return i.newIndexSummary(manifest.DocumentCount, manifest.GeneratedAt), nil
	}

	existing, err := i.store.Load(ctx, indexPath)
	if err != nil || !i.canReuseExistingIndex(existing) {
		docs, err := i.buildDocuments(ctx, files, nil, hashes)
		if err != nil {
			return nil, err
		}
		index := i.newIndex(docs)
		if err := i.store.Save(ctx, indexPath, index); err != nil {
			return nil, err
		}
		return index, nil
	}

	previous := make(map[string]Document, len(existing.Documents))
	for _, doc := range existing.Documents {
		previous[doc.Path] = doc
	}
	docs, err := i.buildDocuments(ctx, files, previous, hashes)
	if err != nil {
		return nil, err
	}
	index := i.newIndex(docs)
	if sameDocuments(existing.Documents, docs) {
		return existing, nil
	}
	if err := i.store.Save(ctx, indexPath, index); err != nil {
		return nil, err
	}
	return index, nil
}

func (i *Indexer) canReuseExistingIndex(index *Index) bool {
	return index != nil &&
		index.Version == IndexVersion &&
		index.Embedding == i.embedding
}

func (i *Indexer) canReuseManifest(manifest *IndexManifest) bool {
	return manifest != nil &&
		manifest.Version == IndexVersion &&
		manifest.Embedding == i.embedding
}

func (i *Indexer) readTextFiles(ctx context.Context, source string) ([]TextFile, error) {
	files, err := i.source.ReadTextFiles(ctx, source)
	if err != nil {
		return nil, err
	}
	if i.ignoredPath == "" {
		return files, nil
	}
	return slices.DeleteFunc(files, func(file TextFile) bool { return file.Path == i.ignoredPath }), nil
}

const (
	defaultEmbedBatchSize = 32
	defaultEmbedWorkers   = 4
)

type embedWork struct {
	index int
	file  TextFile
	hash  string
}

func (i *Indexer) buildDocuments(
	ctx context.Context,
	files []TextFile,
	previous map[string]Document,
	hashes map[string]string,
) ([]Document, error) {
	docs := make([]Document, len(files))
	var workItems []embedWork

	for idx, file := range files {
		if strings.TrimSpace(file.Content) == "" {
			continue
		}
		hash := hashes[file.Path]
		previousDoc, hasPrevious := previous[file.Path]
		canReuse := hasPrevious &&
			previousDoc.ContentHash == hash &&
			len(previousDoc.Embedding) > 0 &&
			previousDoc.Terms != nil
		if canReuse {
			docs[idx] = previousDoc
			if i.progress != nil {
				i.progress(IndexProgress{Current: idx + 1, Total: len(files), Path: file.Path, Cached: true})
			}
			continue
		}
		workItems = append(workItems, embedWork{index: idx, file: file, hash: hash})
	}

	if err := i.embedWorkItems(ctx, files, docs, workItems); err != nil {
		return nil, err
	}

	docs = slices.DeleteFunc(docs, func(d Document) bool { return d.Path == "" })
	slices.SortFunc(docs, func(a, b Document) int { return cmp.Compare(a.Path, b.Path) })
	return docs, nil
}

func (i *Indexer) embedWorkItems(ctx context.Context, files []TextFile, docs []Document, workItems []embedWork) error {
	if len(workItems) == 0 {
		return nil
	}

	chunkSize := defaultEmbedBatchSize
	workers := defaultEmbedWorkers

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(workers)
	for start := 0; start < len(workItems); start += chunkSize {
		end := min(start+chunkSize, len(workItems))
		chunk := workItems[start:end]
		g.Go(func() error {
			return i.embedChunk(ctx, files, docs, chunk)
		})
	}
	return g.Wait()
}

func (i *Indexer) embedChunk(ctx context.Context, files []TextFile, docs []Document, chunk []embedWork) error {
	texts := make([]string, 0, len(chunk))
	pending := make([]*Document, 0, len(chunk))
	for _, w := range chunk {
		indexTitle := shortDocumentTitle(Document{Path: w.file.Path, Content: w.file.Content})
		terms := countTerms(i.tokenizer.Tokenize(indexTitle + " " + indexTitle + " " + indexTitle + " " + w.file.Content))
		doc := &docs[w.index]
		doc.Path = w.file.Path
		doc.Content = w.file.Content
		doc.ContentHash = w.hash
		doc.Terms = terms
		doc.Length = termCount(terms)
		texts = append(texts, doc.Content)
		pending = append(pending, doc)
	}

	embeddings, err := i.embedBatchWithFallback(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed batch: %w", err)
	}
	if len(embeddings) != len(texts) {
		return fmt.Errorf("embed batch: expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	for j, doc := range pending {
		doc.Embedding = embeddings[j]
		if i.progress != nil {
			idx := slices.IndexFunc(files, func(f TextFile) bool { return f.Path == doc.Path })
			if idx >= 0 {
				i.progress(IndexProgress{Current: idx + 1, Total: len(files), Path: doc.Path})
			}
		}
	}
	return nil
}

func (i *Indexer) embedBatchWithFallback(ctx context.Context, texts []string) ([][]float64, error) {
	embeddings, err := i.embedder.EmbedDocuments(ctx, texts)
	if err == nil {
		return embeddings, nil
	}
	if !errors.Is(err, ErrEmbedTextTooLarge) {
		return nil, err
	}

	out := make([][]float64, len(texts))
	for idx, text := range texts {
		embedding, embedErr := i.embedText(ctx, text)
		if embedErr != nil {
			return nil, embedErr
		}
		out[idx] = embedding
	}
	return out, nil
}

func (i *Indexer) embedText(ctx context.Context, text string) ([]float64, error) {
	embeddings, err := i.embedder.EmbedDocuments(ctx, []string{text})
	if err == nil {
		return embeddings[0], nil
	}
	if !errors.Is(err, ErrEmbedTextTooLarge) {
		return nil, err
	}

	first, second, ok := splitTextInHalf(text)
	if !ok {
		return nil, err
	}
	firstEmbedding, err := i.embedText(ctx, first)
	if err != nil {
		return nil, err
	}
	secondEmbedding, err := i.embedText(ctx, second)
	if err != nil {
		return nil, err
	}
	return averageEmbeddings(firstEmbedding, secondEmbedding), nil
}

const minSplitRunes = 20

func splitTextInHalf(text string) (first string, second string, ok bool) {
	runes := []rune(text)
	if len(runes) < minSplitRunes*2 {
		return "", "", false
	}

	mid := len(runes) / 2
	cut := mid
	for offset := range mid {
		if unicode.IsSpace(runes[mid+offset]) {
			cut = mid + offset
			break
		}
		if unicode.IsSpace(runes[mid-offset]) {
			cut = mid - offset
			break
		}
	}

	first = strings.TrimSpace(string(runes[:cut]))
	second = strings.TrimSpace(string(runes[cut:]))
	if first == "" || second == "" {
		return "", "", false
	}
	return first, second, true
}

func averageEmbeddings(a []float64, b []float64) []float64 {
	out := make([]float64, len(a))
	for i, av := range a {
		out[i] = (av + b[i]) / 2
	}
	return out
}

func (i *Indexer) newIndex(docs []Document) *Index {
	return &Index{
		Version:       IndexVersion,
		Embedding:     i.embedding,
		DocumentCount: len(docs),
		GeneratedAt:   time.Now().UTC(),
		Documents:     docs,
	}
}

func (i *Indexer) newIndexSummary(documentCount int, generatedAt time.Time) *Index {
	return &Index{
		Version:       IndexVersion,
		Embedding:     i.embedding,
		DocumentCount: documentCount,
		GeneratedAt:   generatedAt,
	}
}

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
	embedding.Model = strings.TrimSpace(embedding.Model)
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
	if recorded.Model != "" && configured.Model != "" && recorded.Model != configured.Model {
		return fmt.Errorf(
			"index was built with embedding model %s; search with --embedder-model %s or rebuild index",
			recorded.Model,
			recorded.Model,
		)
	}
	if recorded.QueryPrefix != configured.QueryPrefix || recorded.DocumentPrefix != configured.DocumentPrefix {
		return fmt.Errorf(
			"index was built with query prefix %q and document prefix %q; search with --embedder-query-prefix %q --embedder-document-prefix %q or rebuild index",
			recorded.QueryPrefix,
			recorded.DocumentPrefix,
			recorded.QueryPrefix,
			recorded.DocumentPrefix,
		)
	}
	return nil
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

	df := make(map[string]int, len(queryTerms))
	for _, term := range queryTerms {
		for _, doc := range docs {
			if doc.Terms[term] > 0 {
				df[term]++
			}
		}
	}

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
	df := make(map[string]int, len(queryTerms))
	for _, term := range queryTerms {
		for _, doc := range docs {
			if doc.Terms[term] > 0 {
				df[term]++
			}
		}
	}

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

func contentHash(content string) string {
	sum := sha256.Sum256([]byte(content))
	return fmt.Sprintf("%x", sum)
}

func contentHashes(files []TextFile) map[string]string {
	hashes := make(map[string]string, len(files))
	for _, file := range files {
		hashes[file.Path] = contentHash(file.Content)
	}
	return hashes
}

func manifestMatchesFiles(manifest *IndexManifest, files []TextFile, hashes map[string]string) bool {
	if manifest == nil || len(manifest.Documents) != len(files) {
		return false
	}
	for _, file := range files {
		doc, ok := manifest.Documents[file.Path]
		if !ok || doc.ContentHash != hashes[file.Path] {
			return false
		}
	}
	return true
}

func sameDocuments(a []Document, b []Document) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Path != b[i].Path || a[i].ContentHash != b[i].ContentHash {
			return false
		}
	}
	return true
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
