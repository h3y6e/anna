package core

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io/fs"
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
	if errors.Is(err, fs.ErrNotExist) || errors.Is(err, ErrIndexVersionMismatch) {
		return i.rebuild(ctx, files, hashes, indexPath)
	}
	if err != nil {
		return nil, err
	}
	if manifest.Embedding != i.embedding {
		return i.rebuild(ctx, files, hashes, indexPath)
	}
	if manifestMatchesFiles(manifest, files, hashes) {
		return i.newIndexSummary(manifest.DocumentCount, manifest.GeneratedAt), nil
	}

	existing, err := i.store.Load(ctx, indexPath)
	if err != nil {
		return nil, err
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

func (i *Indexer) rebuild(ctx context.Context, files []TextFile, hashes map[string]string, indexPath string) (*Index, error) {
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
	embedBatchSize = 32
	embedWorkers   = 4
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

	if err := i.embedWorkItems(ctx, docs, workItems); err != nil {
		return nil, err
	}

	docs = slices.DeleteFunc(docs, func(d Document) bool { return d.Path == "" })
	slices.SortFunc(docs, func(a, b Document) int { return cmp.Compare(a.Path, b.Path) })
	return docs, nil
}

func (i *Indexer) embedWorkItems(ctx context.Context, docs []Document, workItems []embedWork) error {
	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(embedWorkers)
	for chunk := range slices.Chunk(workItems, embedBatchSize) {
		g.Go(func() error {
			return i.embedChunk(ctx, docs, chunk)
		})
	}
	return g.Wait()
}

func (i *Indexer) embedChunk(ctx context.Context, docs []Document, chunk []embedWork) error {
	texts := make([]string, 0, len(chunk))
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
	}

	embeddings, err := i.embedBatchWithFallback(ctx, texts)
	if err != nil {
		return err
	}

	for j, w := range chunk {
		docs[w.index].Embedding = embeddings[j]
		if i.progress != nil {
			i.progress(IndexProgress{Current: w.index + 1, Total: len(docs), Path: w.file.Path})
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
		Embedding:     i.embedding,
		DocumentCount: len(docs),
		GeneratedAt:   time.Now().UTC(),
		Documents:     docs,
	}
}

func (i *Indexer) newIndexSummary(documentCount int, generatedAt time.Time) *Index {
	return &Index{
		Embedding:     i.embedding,
		DocumentCount: documentCount,
		GeneratedAt:   generatedAt,
	}
}

func contentHash(content string) string {
	sum := sha256.Sum256([]byte(content))
	return hex.EncodeToString(sum[:])
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
