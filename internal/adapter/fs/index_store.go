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
	indexEmbeddingModelKey = []byte("embedding_model")
	indexQueryPrefixKey    = []byte("query_prefix")
	indexDocumentPrefixKey = []byte("document_prefix")
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

		if err := meta.Put(indexVersionKey, []byte(strconv.Itoa(core.IndexVersion))); err != nil {
			return fmt.Errorf("write index version: %w", err)
		}
		if err := putEmbeddingProfile(meta, index.Embedding); err != nil {
			return err
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
		meta, err := readMeta(tx)
		if err != nil {
			return err
		}
		bucket := tx.Bucket(indexManifestBucket)
		if bucket == nil {
			return fmt.Errorf("index manifest bucket is missing")
		}
		manifest.Embedding = meta.embedding
		manifest.DocumentCount = meta.documentCount
		manifest.GeneratedAt = meta.generatedAt

		manifest.Documents = make(map[string]core.DocumentManifest, bucket.Stats().KeyN)
		if err := bucket.ForEach(func(key []byte, value []byte) error {
			manifest.Documents[string(key)] = core.DocumentManifest{ContentHash: string(value)}
			return nil
		}); err != nil {
			return fmt.Errorf("read index manifest: %w", err)
		}
		return nil
	}); err != nil {
		return nil, err
	}
	return manifest, nil
}

func (IndexStore) LoadSearchDocuments(
	ctx context.Context,
	path string,
	terms []string,
	withEmbedding bool,
) (core.EmbeddingProfile, []core.Document, error) {
	if err := ctx.Err(); err != nil {
		return core.EmbeddingProfile{}, nil, err
	}
	db, err := bolt.Open(path, 0o600, &bolt.Options{ReadOnly: true, Timeout: time.Second})
	if err != nil {
		return core.EmbeddingProfile{}, nil, fmt.Errorf("open index: %w", err)
	}
	defer db.Close()

	var embedding core.EmbeddingProfile
	var docs []core.Document
	err = db.View(func(tx *bolt.Tx) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		meta, err := readMeta(tx)
		if err != nil {
			return err
		}
		embedding = meta.embedding

		documentInfo := tx.Bucket(indexDocumentInfoBucket)
		postingsBucket := tx.Bucket(indexPostingsBucket)
		embeddings := tx.Bucket(indexEmbeddingsBucket)
		if documentInfo == nil || postingsBucket == nil || embeddings == nil {
			return fmt.Errorf("index optimized search buckets are missing; rebuild index")
		}
		termFrequencies, err := readQueryTermFrequencies(postingsBucket, terms)
		if err != nil {
			return err
		}
		docs = make([]core.Document, 0, documentInfo.Stats().KeyN)
		if err := documentInfo.ForEach(func(key []byte, value []byte) error {
			info, err := decodeIndexDocumentInfo(value)
			if err != nil {
				return fmt.Errorf("decode document info %s: %w", string(key), err)
			}
			doc := core.Document{
				Path:        info.Path,
				Content:     info.Content,
				ContentHash: info.ContentHash,
				Terms:       termFrequencies[info.Path],
				Length:      info.Length,
			}
			if withEmbedding {
				doc.Embedding, err = decodeEmbedding(embeddings.Get(key))
				if err != nil {
					return fmt.Errorf("decode embedding %s: %w", info.Path, err)
				}
			}
			docs = append(docs, doc)
			return nil
		}); err != nil {
			return fmt.Errorf("read index documents: %w", err)
		}
		return nil
	})
	return embedding, docs, err
}

func putEmbeddingProfile(meta *bolt.Bucket, embedding core.EmbeddingProfile) error {
	if err := meta.Put(indexEmbeddingModelKey, []byte(embedding.Model)); err != nil {
		return fmt.Errorf("write index embedding model: %w", err)
	}
	if err := meta.Put(indexQueryPrefixKey, []byte(embedding.QueryPrefix)); err != nil {
		return fmt.Errorf("write index query prefix: %w", err)
	}
	if err := meta.Put(indexDocumentPrefixKey, []byte(embedding.DocumentPrefix)); err != nil {
		return fmt.Errorf("write index document prefix: %w", err)
	}
	return nil
}

type indexMeta struct {
	embedding     core.EmbeddingProfile
	documentCount int
	generatedAt   time.Time
}

func readMeta(tx *bolt.Tx) (indexMeta, error) {
	meta := tx.Bucket(indexMetaBucket)
	if meta == nil {
		return indexMeta{}, fmt.Errorf("index metadata bucket is missing")
	}
	version, err := strconv.Atoi(string(meta.Get(indexVersionKey)))
	if err != nil {
		return indexMeta{}, fmt.Errorf("decode index version: %w", err)
	}
	if version != core.IndexVersion {
		return indexMeta{}, fmt.Errorf("index version %d, want %d: %w", version, core.IndexVersion, core.ErrIndexVersionMismatch)
	}
	documentCount, err := strconv.Atoi(string(meta.Get(indexDocumentCountKey)))
	if err != nil {
		return indexMeta{}, fmt.Errorf("decode index document count: %w", err)
	}
	generatedAt, err := time.Parse(time.RFC3339Nano, string(meta.Get(indexGeneratedAtKey)))
	if err != nil {
		return indexMeta{}, fmt.Errorf("decode index generated time: %w", err)
	}
	return indexMeta{
		embedding: core.EmbeddingProfile{
			Model:          string(meta.Get(indexEmbeddingModelKey)),
			QueryPrefix:    string(meta.Get(indexQueryPrefixKey)),
			DocumentPrefix: string(meta.Get(indexDocumentPrefixKey)),
		},
		documentCount: documentCount,
		generatedAt:   generatedAt,
	}, nil
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
		meta, err := readMeta(tx)
		if err != nil {
			return err
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

		index.Embedding = meta.embedding
		index.DocumentCount = meta.documentCount
		index.GeneratedAt = meta.generatedAt

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
