package fs

import (
	"bytes"
	"cmp"
	"context"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"time"

	"github.com/h3y6e/anna/internal/core"
	bolt "go.etcd.io/bbolt"
)

var (
	indexMetaBucket         = []byte("meta")
	indexManifestBucket     = []byte("manifest")
	indexDocumentInfoBucket = []byte("document_info")
	indexPostingsBucket     = []byte("postings")
	indexEmbeddingsBucket   = []byte("embeddings")
)

var (
	indexVersionKey        = []byte("version")
	indexSourceKey         = []byte("source")
	indexEmbeddingModelKey = []byte("embedding_model")
	indexGeneratedAtKey    = []byte("generated_at")
	indexDocumentCountKey  = []byte("document_count")
	indexTotalLengthKey    = []byte("total_length")
)

type indexDocumentInfo struct {
	Path        string
	Content     string
	ContentHash string
	Length      int
}

type indexPosting struct {
	Path      string
	Frequency int
}

type IndexStore struct{}

func (IndexStore) Save(ctx context.Context, path string, index *core.Index) (err error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	if index == nil {
		return fmt.Errorf("index is required")
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create index directory: %w", err)
	}

	file, err := os.CreateTemp(dir, ".anna-memory-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary index: %w", err)
	}
	tmpPath := file.Name()
	if err := file.Close(); err != nil {
		return fmt.Errorf("close temporary index: %w", err)
	}
	defer func() {
		if err == nil {
			return
		}
		if removeErr := os.Remove(tmpPath); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			err = errors.Join(err, fmt.Errorf("remove temporary index: %w", removeErr))
		}
	}()

	db, err := bolt.Open(tmpPath, 0o600, &bolt.Options{Timeout: time.Second})
	if err != nil {
		return fmt.Errorf("open temporary index database: %w", err)
	}
	if err := saveIndexToDB(ctx, db, index); err != nil {
		if closeErr := db.Close(); closeErr != nil {
			return errors.Join(err, fmt.Errorf("close temporary index database: %w", closeErr))
		}
		return err
	}
	if err := db.Close(); err != nil {
		return fmt.Errorf("close temporary index database: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("replace index: %w", err)
	}
	return nil
}

func saveIndexToDB(ctx context.Context, db *bolt.DB, index *core.Index) error {
	return db.Update(func(tx *bolt.Tx) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		meta, err := tx.CreateBucket(indexMetaBucket)
		if err != nil {
			return fmt.Errorf("create index metadata bucket: %w", err)
		}
		documentInfo, err := tx.CreateBucket(indexDocumentInfoBucket)
		if err != nil {
			return fmt.Errorf("create index document info bucket: %w", err)
		}
		manifest, err := tx.CreateBucket(indexManifestBucket)
		if err != nil {
			return fmt.Errorf("create index manifest bucket: %w", err)
		}
		postingsBucket, err := tx.CreateBucket(indexPostingsBucket)
		if err != nil {
			return fmt.Errorf("create index postings bucket: %w", err)
		}
		embeddings, err := tx.CreateBucket(indexEmbeddingsBucket)
		if err != nil {
			return fmt.Errorf("create index embeddings bucket: %w", err)
		}

		if err := meta.Put(indexVersionKey, []byte(strconv.Itoa(index.Version))); err != nil {
			return fmt.Errorf("write index version: %w", err)
		}
		if err := meta.Put(indexSourceKey, []byte(index.Source)); err != nil {
			return fmt.Errorf("write index source: %w", err)
		}
		if err := meta.Put(indexEmbeddingModelKey, []byte(index.EmbeddingModel)); err != nil {
			return fmt.Errorf("write index embedding model: %w", err)
		}
		if err := meta.Put(indexGeneratedAtKey, []byte(index.GeneratedAt.Format(time.RFC3339Nano))); err != nil {
			return fmt.Errorf("write index generated time: %w", err)
		}
		if err := meta.Put(indexDocumentCountKey, []byte(strconv.Itoa(len(index.Documents)))); err != nil {
			return fmt.Errorf("write index document count: %w", err)
		}

		var totalLength int
		postings := make(map[string][]indexPosting)
		for _, doc := range index.Documents {
			totalLength += doc.Length

			data, err := encodeIndexDocumentInfo(indexDocumentInfo{
				Path:        doc.Path,
				Content:     doc.Content,
				ContentHash: doc.ContentHash,
				Length:      doc.Length,
			})
			if err != nil {
				return fmt.Errorf("encode document info %s: %w", doc.Path, err)
			}
			if err := documentInfo.Put([]byte(doc.Path), data); err != nil {
				return fmt.Errorf("write document info %s: %w", doc.Path, err)
			}
			if err := manifest.Put([]byte(doc.Path), []byte(doc.ContentHash)); err != nil {
				return fmt.Errorf("write document manifest %s: %w", doc.Path, err)
			}

			data = encodeEmbedding(doc.Embedding)
			if err := embeddings.Put([]byte(doc.Path), data); err != nil {
				return fmt.Errorf("write embedding %s: %w", doc.Path, err)
			}

			for term, frequency := range doc.Terms {
				if frequency <= 0 {
					continue
				}
				postings[term] = append(postings[term], indexPosting{Path: doc.Path, Frequency: frequency})
			}
		}
		if err := meta.Put(indexTotalLengthKey, []byte(strconv.Itoa(totalLength))); err != nil {
			return fmt.Errorf("write index total length: %w", err)
		}
		for term, termPostings := range postings {
			slices.SortFunc(termPostings, func(a, b indexPosting) int { return cmp.Compare(a.Path, b.Path) })
			data, err := encodePostings(termPostings)
			if err != nil {
				return fmt.Errorf("encode postings %s: %w", term, err)
			}
			if err := postingsBucket.Put([]byte(term), data); err != nil {
				return fmt.Errorf("write postings %s: %w", term, err)
			}
		}
		return nil
	})
}

func (IndexStore) LoadManifest(ctx context.Context, path string) (*core.IndexManifest, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	db, err := bolt.Open(path, 0o600, &bolt.Options{ReadOnly: true, Timeout: time.Second})
	if err != nil {
		return nil, fmt.Errorf("open index: %w", err)
	}
	defer db.Close()

	manifest := &core.IndexManifest{}
	if err := db.View(func(tx *bolt.Tx) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		meta := tx.Bucket(indexMetaBucket)
		if meta == nil {
			return fmt.Errorf("index metadata bucket is missing")
		}
		bucket := tx.Bucket(indexManifestBucket)
		if bucket == nil {
			return fmt.Errorf("index manifest bucket is missing")
		}

		version, err := strconv.Atoi(string(meta.Get(indexVersionKey)))
		if err != nil {
			return fmt.Errorf("decode index version: %w", err)
		}
		manifest.Version = version
		manifest.Source = string(meta.Get(indexSourceKey))
		manifest.EmbeddingModel = string(meta.Get(indexEmbeddingModelKey))
		if count := meta.Get(indexDocumentCountKey); len(count) > 0 {
			parsed, err := strconv.Atoi(string(count))
			if err != nil {
				return fmt.Errorf("decode index document count: %w", err)
			}
			manifest.DocumentCount = parsed
		}
		if generatedAt := meta.Get(indexGeneratedAtKey); len(generatedAt) > 0 {
			parsed, err := time.Parse(time.RFC3339Nano, string(generatedAt))
			if err != nil {
				return fmt.Errorf("decode index generated time: %w", err)
			}
			manifest.GeneratedAt = parsed
		}

		manifest.Documents = make(map[string]core.DocumentManifest, bucket.Stats().KeyN)
		if err := bucket.ForEach(func(key []byte, value []byte) error {
			manifest.Documents[string(key)] = core.DocumentManifest{ContentHash: string(value)}
			return nil
		}); err != nil {
			return fmt.Errorf("read index manifest: %w", err)
		}
		if manifest.DocumentCount == 0 {
			manifest.DocumentCount = len(manifest.Documents)
		}
		return nil
	}); err != nil {
		return nil, err
	}
	if manifest.Version != core.IndexVersion {
		return nil, fmt.Errorf("unsupported index version %d", manifest.Version)
	}
	return manifest, nil
}

func (IndexStore) Search(
	ctx context.Context,
	path string,
	query string,
	limit int,
	embedder core.Embedder,
	tokenizer core.Tokenizer,
	embeddingModel string,
	mode core.SearchMode,
) ([]core.SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := mode.Validate(); err != nil {
		return nil, err
	}
	if mode.RequiresEmbedding() && embedder == nil {
		return nil, fmt.Errorf("embedder is required")
	}
	if tokenizer == nil {
		return nil, fmt.Errorf("tokenizer is required")
	}

	db, err := bolt.Open(path, 0o600, &bolt.Options{ReadOnly: true, Timeout: time.Second})
	if err != nil {
		return nil, fmt.Errorf("open index: %w", err)
	}
	defer db.Close()

	if err := validateSearchIndex(ctx, db, embeddingModel, mode); err != nil {
		return nil, err
	}
	var queryEmbedding []float64
	if mode.RequiresEmbedding() {
		var err error
		queryEmbedding, err = embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embed query: %w", err)
		}
		if len(queryEmbedding) == 0 {
			return nil, fmt.Errorf("query embedding is empty")
		}
	}
	queryTerms, err := tokenizer.TokenizeQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("tokenize query: %w", err)
	}
	queryTerms = uniqueTokens(queryTerms)

	index := &core.Index{Version: core.IndexVersion}
	if err := db.View(func(tx *bolt.Tx) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		documentInfo := tx.Bucket(indexDocumentInfoBucket)
		postingsBucket := tx.Bucket(indexPostingsBucket)
		if documentInfo == nil || postingsBucket == nil {
			return fmt.Errorf("index optimized search buckets are missing; rebuild index")
		}
		embeddings := tx.Bucket(indexEmbeddingsBucket)
		if mode.RequiresEmbedding() && embeddings == nil {
			return fmt.Errorf("index optimized search buckets are missing; rebuild index")
		}

		termFrequencies, err := readQueryTermFrequencies(postingsBucket, queryTerms)
		if err != nil {
			return err
		}
		index.Documents = make([]core.Document, 0, documentInfo.Stats().KeyN)
		if err := documentInfo.ForEach(func(key []byte, value []byte) error {
			info, err := decodeIndexDocumentInfo(value)
			if err != nil {
				return fmt.Errorf("decode document info %s: %w", string(key), err)
			}
			var embedding []float64
			if mode.RequiresEmbedding() {
				var err error
				embedding, err = decodeEmbedding(embeddings.Get(key))
				if err != nil {
					return fmt.Errorf("decode embedding %s: %w", info.Path, err)
				}
				if len(embedding) == 0 {
					return fmt.Errorf("index document %s has no embedding; rebuild index", info.Path)
				}
				if len(embedding) != len(queryEmbedding) {
					return fmt.Errorf(
						"index document %s embedding dimensions %d do not match query dimensions %d; rebuild index",
						info.Path,
						len(embedding),
						len(queryEmbedding),
					)
				}
			}
			index.Documents = append(index.Documents, core.Document{
				Path:        info.Path,
				Content:     info.Content,
				ContentHash: info.ContentHash,
				Terms:       termFrequencies[info.Path],
				Length:      info.Length,
				Embedding:   embedding,
			})
			return nil
		}); err != nil {
			return fmt.Errorf("read index documents: %w", err)
		}
		return nil
	}); err != nil {
		return nil, err
	}
	slices.SortFunc(index.Documents, func(a, b core.Document) int { return cmp.Compare(a.Path, b.Path) })
	return core.SearchTokenized(index, query, queryTerms, queryEmbedding, limit, mode)
}

func validateSearchIndex(ctx context.Context, db *bolt.DB, embeddingModel string, mode core.SearchMode) error {
	return db.View(func(tx *bolt.Tx) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		meta := tx.Bucket(indexMetaBucket)
		if meta == nil {
			return fmt.Errorf("index metadata bucket is missing")
		}
		version, err := strconv.Atoi(string(meta.Get(indexVersionKey)))
		if err != nil {
			return fmt.Errorf("decode index version: %w", err)
		}
		if version != core.IndexVersion {
			return fmt.Errorf("unsupported index version %d", version)
		}
		if mode.RequiresEmbedding() {
			indexEmbeddingModel := string(meta.Get(indexEmbeddingModelKey))
			if indexEmbeddingModel != "" && embeddingModel != "" && indexEmbeddingModel != embeddingModel {
				return fmt.Errorf(
					"index was built with embedding model %s; search with --embedding-model %s or rebuild index",
					indexEmbeddingModel,
					indexEmbeddingModel,
				)
			}
		}
		return nil
	})
}

func readQueryTermFrequencies(postingsBucket *bolt.Bucket, queryTerms []string) (map[string]map[string]int, error) {
	termFrequencies := make(map[string]map[string]int)
	for _, term := range queryTerms {
		data := postingsBucket.Get([]byte(term))
		if len(data) == 0 {
			continue
		}
		postings, err := decodePostings(data)
		if err != nil {
			return nil, fmt.Errorf("decode postings %s: %w", term, err)
		}
		for _, posting := range postings {
			if posting.Frequency <= 0 {
				continue
			}
			if termFrequencies[posting.Path] == nil {
				termFrequencies[posting.Path] = make(map[string]int, len(queryTerms))
			}
			termFrequencies[posting.Path][term] = posting.Frequency
		}
	}
	return termFrequencies, nil
}

func (IndexStore) Load(ctx context.Context, path string) (*core.Index, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	db, err := bolt.Open(path, 0o600, &bolt.Options{ReadOnly: true, Timeout: time.Second})
	if err != nil {
		return nil, fmt.Errorf("open index: %w", err)
	}
	defer db.Close()

	var index core.Index
	if err := db.View(func(tx *bolt.Tx) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		meta := tx.Bucket(indexMetaBucket)
		if meta == nil {
			return fmt.Errorf("index metadata bucket is missing")
		}
		documentInfo := tx.Bucket(indexDocumentInfoBucket)
		if documentInfo == nil {
			return fmt.Errorf("index document info bucket is missing")
		}
		manifest := tx.Bucket(indexManifestBucket)
		if manifest == nil {
			return fmt.Errorf("index manifest bucket is missing")
		}
		postingsBucket := tx.Bucket(indexPostingsBucket)
		if postingsBucket == nil {
			return fmt.Errorf("index postings bucket is missing")
		}
		embeddings := tx.Bucket(indexEmbeddingsBucket)
		if embeddings == nil {
			return fmt.Errorf("index embeddings bucket is missing")
		}

		version, err := strconv.Atoi(string(meta.Get(indexVersionKey)))
		if err != nil {
			return fmt.Errorf("decode index version: %w", err)
		}
		index.Version = version
		index.Source = string(meta.Get(indexSourceKey))
		index.EmbeddingModel = string(meta.Get(indexEmbeddingModelKey))
		if count := meta.Get(indexDocumentCountKey); len(count) > 0 {
			parsed, err := strconv.Atoi(string(count))
			if err != nil {
				return fmt.Errorf("decode index document count: %w", err)
			}
			index.DocumentCount = parsed
		}
		if generatedAt := meta.Get(indexGeneratedAtKey); len(generatedAt) > 0 {
			parsed, err := time.Parse(time.RFC3339Nano, string(generatedAt))
			if err != nil {
				return fmt.Errorf("decode index generated time: %w", err)
			}
			index.GeneratedAt = parsed
		}

		documentIndexes := make(map[string]int, documentInfo.Stats().KeyN)
		index.Documents = make([]core.Document, 0, documentInfo.Stats().KeyN)
		if err := documentInfo.ForEach(func(key []byte, value []byte) error {
			info, err := decodeIndexDocumentInfo(value)
			if err != nil {
				return fmt.Errorf("decode document info %s: %w", string(key), err)
			}
			embedding, err := decodeEmbedding(embeddings.Get(key))
			if err != nil {
				return fmt.Errorf("decode embedding %s: %w", info.Path, err)
			}
			documentIndexes[info.Path] = len(index.Documents)
			index.Documents = append(index.Documents, core.Document{
				Path:        info.Path,
				Content:     info.Content,
				ContentHash: info.ContentHash,
				Length:      info.Length,
				Embedding:   embedding,
			})
			return nil
		}); err != nil {
			return fmt.Errorf("read index documents: %w", err)
		}
		if err := postingsBucket.ForEach(func(key []byte, value []byte) error {
			postings, err := decodePostings(value)
			if err != nil {
				return fmt.Errorf("decode postings %s: %w", string(key), err)
			}
			term := string(key)
			for _, posting := range postings {
				idx, ok := documentIndexes[posting.Path]
				if !ok || posting.Frequency <= 0 {
					continue
				}
				if index.Documents[idx].Terms == nil {
					index.Documents[idx].Terms = make(map[string]int)
				}
				index.Documents[idx].Terms[term] = posting.Frequency
			}
			return nil
		}); err != nil {
			return fmt.Errorf("read index postings: %w", err)
		}
		return nil
	}); err != nil {
		return nil, err
	}
	if index.Version != core.IndexVersion {
		return nil, fmt.Errorf("unsupported index version %d", index.Version)
	}
	if index.DocumentCount == 0 {
		index.DocumentCount = len(index.Documents)
	}
	slices.SortFunc(index.Documents, func(a, b core.Document) int { return cmp.Compare(a.Path, b.Path) })
	return &index, nil
}

func encodeIndexDocumentInfo(info indexDocumentInfo) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(info); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeIndexDocumentInfo(data []byte) (indexDocumentInfo, error) {
	var info indexDocumentInfo
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&info); err != nil {
		return indexDocumentInfo{}, err
	}
	return info, nil
}

func encodePostings(postings []indexPosting) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(postings); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodePostings(data []byte) ([]indexPosting, error) {
	var postings []indexPosting
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&postings); err != nil {
		return nil, err
	}
	return postings, nil
}

func encodeEmbedding(embedding []float64) []byte {
	data := make([]byte, len(embedding)*8)
	for i, value := range embedding {
		binary.LittleEndian.PutUint64(data[i*8:], math.Float64bits(value))
	}
	return data
}

func decodeEmbedding(data []byte) ([]float64, error) {
	if len(data)%8 != 0 {
		return nil, fmt.Errorf("embedding byte length %d is not a multiple of 8", len(data))
	}
	embedding := make([]float64, len(data)/8)
	for i := range embedding {
		embedding[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return embedding, nil
}

func uniqueTokens(tokens []string) []string {
	seen := make(map[string]bool, len(tokens))
	unique := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if seen[token] {
			continue
		}
		seen[token] = true
		unique = append(unique, token)
	}
	return unique
}
