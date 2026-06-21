package core

import (
	"cmp"
	"context"
	"crypto/sha256"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"time"

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

	embeddingModel string
	progress       func(IndexProgress)
}

type IndexBuildOptions struct {
	Rebuild bool
}

func NewIndexer(source TextSource, store IndexStore, embedder Embedder, tokenizer Tokenizer) *Indexer {
	return &Indexer{source: source, store: store, embedder: embedder, tokenizer: tokenizer}
}

func (i *Indexer) WithEmbeddingModel(model string) *Indexer {
	i.embeddingModel = strings.TrimSpace(model)
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

func (i *Indexer) Build(ctx context.Context, source string) (*Index, error) {
	files, err := i.readTextFiles(ctx, source)
	if err != nil {
		return nil, err
	}
	docs, err := i.buildDocuments(ctx, files, nil, nil)
	if err != nil {
		return nil, err
	}
	return i.newIndex(source, docs), nil
}

func (i *Indexer) BuildAndSave(ctx context.Context, source string, indexPath string) (*Index, error) {
	return i.BuildAndSaveWithOptions(ctx, source, indexPath, IndexBuildOptions{})
}

func (i *Indexer) BuildAndSaveWithOptions(
	ctx context.Context,
	source string,
	indexPath string,
	options IndexBuildOptions,
) (*Index, error) {
	if i.store == nil {
		return nil, fmt.Errorf("index store is required")
	}
	if !options.Rebuild {
		index, err := i.BuildIncremental(ctx, source, indexPath)
		if err != nil {
			return nil, err
		}
		return index, nil
	}
	index, err := i.Build(ctx, source)
	if err != nil {
		return nil, err
	}
	if err := i.store.Save(ctx, indexPath, index); err != nil {
		return nil, err
	}
	return index, nil
}

func (i *Indexer) BuildIncremental(ctx context.Context, source string, indexPath string) (*Index, error) {
	if i.store == nil {
		return nil, fmt.Errorf("index store is required")
	}
	files, err := i.readTextFiles(ctx, source)
	if err != nil {
		return nil, err
	}
	hashes := contentHashes(files)
	if manifestStore, ok := i.store.(IndexManifestStore); ok {
		manifest, err := manifestStore.LoadManifest(ctx, indexPath)
		if err == nil && i.canReuseManifest(manifest, source) && manifestMatchesFiles(manifest, files, hashes) {
			return i.newIndexSummary(source, manifest.DocumentCount, manifest.GeneratedAt), nil
		}
	}

	existing, err := i.store.Load(ctx, indexPath)
	if err != nil || !i.canReuseExistingIndex(existing, source) {
		docs, err := i.buildDocuments(ctx, files, nil, hashes)
		if err != nil {
			return nil, err
		}
		index := i.newIndex(source, docs)
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
	index := i.newIndex(source, docs)
	if sameDocuments(existing.Documents, docs) {
		return existing, nil
	}
	if err := i.store.Save(ctx, indexPath, index); err != nil {
		return nil, err
	}
	return index, nil
}

func (i *Indexer) canReuseExistingIndex(index *Index, source string) bool {
	return index != nil &&
		index.Version == IndexVersion &&
		index.Source == source &&
		index.EmbeddingModel == i.embeddingModel
}

func (i *Indexer) canReuseManifest(manifest *IndexManifest, source string) bool {
	return manifest != nil &&
		manifest.Version == IndexVersion &&
		manifest.Source == source &&
		manifest.EmbeddingModel == i.embeddingModel
}

func (i *Indexer) readTextFiles(ctx context.Context, source string) ([]TextFile, error) {
	if source == "" {
		return nil, fmt.Errorf("source is required")
	}
	if i.source == nil {
		return nil, fmt.Errorf("text source is required")
	}
	if i.embedder == nil {
		return nil, fmt.Errorf("embedder is required")
	}
	if i.tokenizer == nil {
		return nil, fmt.Errorf("tokenizer is required")
	}
	files, err := i.source.ReadTextFiles(ctx, source)
	if err != nil {
		return nil, err
	}
	return files, nil
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
		if hash == "" {
			hash = contentHash(file.Content)
		}
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
		tokens, err := i.tokenizer.TokenizeDocument(ctx, indexTitle+" "+indexTitle+" "+indexTitle+" "+w.file.Content)
		if err != nil {
			return fmt.Errorf("tokenize %s: %w", w.file.Path, err)
		}
		terms := countTerms(tokens)
		doc := &docs[w.index]
		doc.Path = w.file.Path
		doc.Content = w.file.Content
		doc.ContentHash = w.hash
		doc.Terms = terms
		doc.Length = termCount(terms)
		texts = append(texts, embeddingInput(*doc))
		pending = append(pending, doc)
	}

	embeddings, err := i.embedder.EmbedBatch(ctx, texts)
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

func (i *Indexer) newIndex(source string, docs []Document) *Index {
	return &Index{
		Version:        IndexVersion,
		Source:         source,
		EmbeddingModel: i.embeddingModel,
		DocumentCount:  len(docs),
		GeneratedAt:    time.Now().UTC(),
		Documents:      docs,
	}
}

func (i *Indexer) newIndexSummary(source string, documentCount int, generatedAt time.Time) *Index {
	return &Index{
		Version:        IndexVersion,
		Source:         source,
		EmbeddingModel: i.embeddingModel,
		DocumentCount:  documentCount,
		GeneratedAt:    generatedAt,
	}
}

type Searcher struct {
	store     IndexStore
	embedder  Embedder
	tokenizer Tokenizer

	embeddingModel string
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

func (s *Searcher) WithEmbeddingModel(model string) *Searcher {
	s.embeddingModel = strings.TrimSpace(model)
	return s
}

func (s *Searcher) SearchFile(
	ctx context.Context,
	indexPath string,
	query string,
	limit int,
	mode SearchMode,
) ([]SearchResult, error) {
	if s.store == nil {
		return nil, fmt.Errorf("index store is required")
	}
	if err := mode.Validate(); err != nil {
		return nil, err
	}
	if mode.RequiresEmbedding() && s.embedder == nil {
		return nil, fmt.Errorf("embedder is required")
	}
	if s.tokenizer == nil {
		return nil, fmt.Errorf("tokenizer is required")
	}
	if store, ok := s.store.(SearchIndexStore); ok {
		return store.Search(ctx, indexPath, query, limit, s.embedder, s.tokenizer, s.embeddingModel, mode)
	}

	index, err := s.store.Load(ctx, indexPath)
	if err != nil {
		return nil, err
	}
	var queryEmbedding []float64
	if mode.RequiresEmbedding() {
		if err := validateSearchEmbeddingModel(index, s.embeddingModel); err != nil {
			return nil, err
		}
		queryEmbedding, err = s.embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embed query: %w", err)
		}
		if err := validateSearchEmbeddings(index, queryEmbedding); err != nil {
			return nil, err
		}
	}
	return Search(ctx, index, query, queryEmbedding, limit, s.tokenizer, mode)
}

func validateSearchEmbeddingModel(index *Index, embeddingModel string) error {
	if index == nil || index.EmbeddingModel == "" || embeddingModel == "" || index.EmbeddingModel == embeddingModel {
		return nil
	}
	return fmt.Errorf(
		"index was built with embedding model %s; search with --embedding-model %s or rebuild index",
		index.EmbeddingModel,
		index.EmbeddingModel,
	)
}

func validateSearchEmbeddings(index *Index, queryEmbedding []float64) error {
	if len(queryEmbedding) == 0 {
		return fmt.Errorf("query embedding is empty")
	}
	if index == nil {
		return nil
	}
	for _, doc := range index.Documents {
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

func Search(
	ctx context.Context,
	index *Index,
	query string,
	queryEmbedding []float64,
	limit int,
	tokenizer Tokenizer,
	mode SearchMode,
) ([]SearchResult, error) {
	if tokenizer == nil {
		return nil, fmt.Errorf("tokenizer is required")
	}
	queryTerms, err := tokenizer.TokenizeQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("tokenize query: %w", err)
	}
	return SearchTokenized(index, query, queryTerms, queryEmbedding, limit, mode)
}

func SearchTokenized(
	index *Index,
	query string,
	queryTerms []string,
	queryEmbedding []float64,
	limit int,
	mode SearchMode,
) ([]SearchResult, error) {
	if err := mode.Validate(); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 10
	}
	if index == nil {
		return nil, nil
	}
	queryTerms = unique(queryTerms)
	if len(index.Documents) == 0 {
		return nil, nil
	}
	if mode == SearchModeRRF {
		return searchRRF(index, query, queryTerms, queryEmbedding, limit), nil
	}

	df := make(map[string]int, len(queryTerms))
	for _, term := range queryTerms {
		for _, doc := range index.Documents {
			if doc.Terms[term] > 0 {
				df[term]++
			}
		}
	}

	docCount := len(index.Documents)
	avgLen := averageLength(index.Documents)
	lowerQuery := strings.ToLower(strings.TrimSpace(query))
	results := make([]SearchResult, 0, len(index.Documents))
	for _, doc := range index.Documents {
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
	return results, nil
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

func searchRRF(index *Index, query string, queryTerms []string, queryEmbedding []float64, limit int) []SearchResult {
	df := make(map[string]int, len(queryTerms))
	for _, term := range queryTerms {
		for _, doc := range index.Documents {
			if doc.Terms[term] > 0 {
				df[term]++
			}
		}
	}

	docCount := len(index.Documents)
	avgLen := averageLength(index.Documents)
	keywordList := make([]rrfScoredDocument, 0, len(index.Documents))
	vectorList := make([]rrfScoredDocument, 0, len(index.Documents))
	for _, doc := range index.Documents {
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

	fused := make(map[string]rrfFusedDocument, len(index.Documents))
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

func embeddingInput(doc Document) string {
	const maxEmbeddingInputRunes = 4000

	runes := []rune(doc.Content)
	if len(runes) > maxEmbeddingInputRunes {
		return string(runes[:maxEmbeddingInputRunes])
	}
	return doc.Content
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
