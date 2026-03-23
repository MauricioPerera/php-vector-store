# PHP Vector Store

Zero-dependency PHP vector database with **BM25 full-text search**, **hybrid search** (vector + text), Matryoshka progressive search, IVF indexing, and Int8 quantization. Pure PHP 8.1+ — no SQLite, no C extensions, no FFI.

```
composer require mauricioperera/php-vector-store
```

## Why

Most vector databases require C extensions (sqlite-vec), external services (Pinecone, Weaviate), or specific runtimes (Python). PHP Vector Store runs anywhere PHP runs — shared hosting, WordPress, Laravel, any framework.

**New in v0.2:** BM25 full-text search, hybrid search fusion (RRF + Weighted), multiple distance metrics, `StoreInterface` for polymorphism, typed models, and a PHPUnit test suite.

## Scaling Guide

| Vectors | Recommended Config | Storage/vec | Total (100K) | Speed |
|---------|-------------------|-------------|-------------|-------|
| <5K | Float32 768d + Matryoshka | 3,072 B | 300 MB | ~3ms |
| 5K-20K | Float32 384d + Matryoshka | 1,536 B | 150 MB | ~1.4ms |
| 20K-100K | Int8 384d + IVF + Matryoshka | **392 B** | **38 MB** | ~5ms |
| 100K-500K | Int8 384d + IVF + Matryoshka | **392 B** | **192 MB** | ~15ms |
| >500K | Use sqlite-vec or external service | — | — | — |

## Quick Start

```php
use PHPVectorStore\VectorStore;
use PHPVectorStore\QuantizedStore;
use PHPVectorStore\IVFIndex;
use PHPVectorStore\HybridSearch;
use PHPVectorStore\HybridMode;
use PHPVectorStore\Distance;
use PHPVectorStore\BM25\Index as BM25Index;

// 1. Vector search
$store = new QuantizedStore( __DIR__ . '/vectors', 384 );
$store->set( 'articles', 'art-1', $embedding, ['title' => 'My Article'] );
$store->flush();

$results = $store->matryoshkaSearch( 'articles', $query, 5, [128, 256, 384] );

// 2. Full-text search (BM25)
$bm25 = new BM25Index();
$bm25->addDocument( 'articles', 'art-1', 'My article about machine learning...' );

$results = $bm25->search( 'articles', 'machine learning', 10 );

// 3. Hybrid search (vector + text combined)
$hybrid = new HybridSearch( $store, $bm25, HybridMode::RRF );
$results = $hybrid->search( 'articles', $query_vector, 'machine learning', 5 );

// 4. Multiple distance metrics
$results = $store->search( 'articles', $query, 5, 0, Distance::Euclidean );
```

## Features

### Vector Storage (Float32 & Int8)

```php
// Full precision: dim x 4 bytes per vector
$store = new VectorStore( '/path', 768 );

// Quantized: dim + 8 bytes per vector (4x smaller)
$q8 = new QuantizedStore( '/path', 384 );
```

Both implement `StoreInterface` — use them interchangeably.

### BM25 Full-Text Search

Okapi BM25 inverted index, collection-aware, with persistence.

```php
use PHPVectorStore\BM25\Index;
use PHPVectorStore\BM25\Config;
use PHPVectorStore\BM25\SimpleTokenizer;

$bm25 = new Index(
    config: new Config( k1: 1.5, b: 0.75 ),
    tokenizer: new SimpleTokenizer(),
);

// Index documents
$bm25->addDocument( 'articles', 'doc-1', 'The quick brown fox...' );
$bm25->addDocument( 'articles', 'doc-2', 'Database systems and SQL...' );

// Search
$results = $bm25->search( 'articles', 'quick fox', 10 );
// [['id' => 'doc-1', 'score' => 1.234, 'rank' => 1], ...]

// Get raw scores (for hybrid fusion)
$scores = $bm25->scoreAll( 'articles', 'quick fox' );
// ['doc-1' => 1.234, 'doc-2' => 0.0]

// Persist to disk
$bm25->save( '/path/vectors', 'articles' );  // writes articles.bm25.bin
$bm25->load( '/path/vectors', 'articles' );  // restores state
```

The `SimpleTokenizer` handles Unicode text with configurable stop words:

```php
// Custom stop words for Spanish
$tokenizer = new SimpleTokenizer(
    stopWords: ['el', 'la', 'los', 'las', 'de', 'en', 'y', 'que', 'es', 'un', 'una'],
    minTokenLength: 2,
);
$bm25 = new Index( tokenizer: $tokenizer );
```

### Hybrid Search

Combines vector similarity with BM25 text relevance using fusion strategies.

```php
use PHPVectorStore\HybridSearch;
use PHPVectorStore\HybridMode;

// RRF fusion (recommended — robust, no tuning needed)
$hybrid = new HybridSearch( $store, $bm25, HybridMode::RRF );
$results = $hybrid->search( 'articles', $vector, 'search text', 5 );

// Weighted fusion (tunable weights)
$hybrid = new HybridSearch( $store, $bm25, HybridMode::Weighted );
$results = $hybrid->search( 'articles', $vector, 'search text', 5, [
    'vectorWeight' => 0.7,
    'textWeight'   => 0.3,
]);

// Multi-collection hybrid
$results = $hybrid->searchAcross(
    ['articles', 'comments'],
    $vector, 'search text', 10,
);
```

**RRF (Reciprocal Rank Fusion):** `score(d) = Σ 1/(k + rank(d))` — combines ranks from both legs without needing score normalization. Best default choice.

**Weighted:** Min-max normalizes both score sets to [0,1], then `combined = w_vec * vecNorm + w_text * textNorm`. Use when you want explicit control over the balance.

### Distance Metrics

```php
use PHPVectorStore\Distance;

// Cosine similarity (default) — best for normalized embeddings
$store->search( 'col', $query, 5, 0, Distance::Cosine );

// Euclidean distance — converted to similarity: 1/(1+dist)
$store->search( 'col', $query, 5, 0, Distance::Euclidean );

// Dot product — for pre-normalized vectors
$store->search( 'col', $query, 5, 0, Distance::DotProduct );

// Manhattan distance — robust to outliers: 1/(1+dist)
$store->search( 'col', $query, 5, 0, Distance::Manhattan );
```

Works with `search()`, `matryoshkaSearch()`, and `searchAcross()` on both VectorStore and QuantizedStore.

### IVF Clustering

K-means partitions vectors into clusters for sub-linear search.

```php
$ivf = new IVFIndex( $store, numClusters: 100, numProbes: 20 );
$ivf->build( 'articles' );
$results = $ivf->search( 'articles', $query, 5 );
$results = $ivf->matryoshkaSearch( 'articles', $query, 5, [128, 256, 384] );
```

Works with both VectorStore and QuantizedStore (via `StoreInterface`).

### Matryoshka Multi-Stage Search

Progressive refinement — each stage narrows candidates before the next.

```php
$store->matryoshkaSearch( 'col', $query, 5, [128, 384, 768] );
```

Speedup: **3-5x** over brute-force. Combined with IVF: **10-15x**.

### StoreInterface

Both `VectorStore` and `QuantizedStore` implement `StoreInterface`:

```php
use PHPVectorStore\StoreInterface;

function buildIndex( StoreInterface $store ): void {
    $ivf = new IVFIndex( $store );
    $ivf->build( 'articles' );
}

// Works with either store
buildIndex( new VectorStore( '/path', 384 ) );
buildIndex( new QuantizedStore( '/path', 384 ) );
```

### Typed Models

```php
use PHPVectorStore\Document;
use PHPVectorStore\SearchResult;

$doc = new Document(
    id: 'doc-1',
    vector: [0.1, 0.2, ...],
    text: 'The quick brown fox...',
    metadata: ['title' => 'My Doc'],
);

$result = new SearchResult(
    id: 'doc-1',
    score: 0.95,
    rank: 1,
    metadata: ['title' => 'My Doc'],
    collection: 'articles',
);
```

### Typed Exceptions

```php
use PHPVectorStore\Exception\VectorStoreException;
use PHPVectorStore\Exception\DimensionMismatchException;
use PHPVectorStore\Exception\CollectionNotFoundException;
```

## Concurrency & Scaling Notes

### File Locking

All `flush()` operations use `flock(LOCK_EX)` to prevent race conditions when multiple PHP processes write to the same collection simultaneously. This ensures atomic writes even under concurrent web requests.

### Dimension Validation

`set()` throws `DimensionMismatchException` if the vector has fewer dimensions than the store was configured with. This catches mismatches early (e.g., passing a 384d vector to a 768d store).

### JSON Manifest Scaling

Each collection stores its ID list and metadata in a `.json` sidecar file. For collections approaching 100K vectors, this manifest can grow large (~10-20 MB). Considerations:

- **Memory**: The entire manifest is loaded into memory on first access to a collection. For 100K vectors with metadata, budget ~50-100 MB of PHP memory.
- **Latency**: JSON decode of a large manifest adds ~50-200ms on first load (cached for subsequent operations within the same request).
- **Mitigation**: Use multiple collections (per entity type) to keep individual manifests small. A collection of 10K vectors has a ~1-2 MB manifest.

For datasets beyond 100K vectors, consider sqlite-vec or an external vector database.

## API Reference

### StoreInterface (VectorStore & QuantizedStore)

```php
// Write
->set( $collection, $id, $vector, $metadata = [] )
->remove( $collection, $id ): bool
->drop( $collection )
->flush()

// Read
->get( $collection, $id ): ?array     // {id, vector, metadata}
->has( $collection, $id ): bool
->count( $collection ): int
->ids( $collection ): string[]
->collections(): string[]
->stats(): array
->dimensions(): int
->directory(): string

// Search
->search( $collection, $query, $limit = 5, $dimSlice = 0, $distance = null )
->matryoshkaSearch( $collection, $query, $limit = 5, $stages = [...], $multiplier = 3, $distance = null )
->searchAcross( $collections, $query, $limit = 5, $dimSlice = 0, $distance = null )

// Import/Export
->import( $collection, $records ): int
->export( $collection ): array
```

### BM25\Index

```php
->addDocument( $collection, $id, $text )
->removeDocument( $collection, $id )
->search( $collection, $query, $limit = 10 ): array
->scoreAll( $collection, $query ): array    // id => score
->count( $collection ): int
->vocabularySize( $collection ): int
->save( $directory, $collection )
->load( $directory, $collection )
->exportState( $collection ): array
->importState( $collection, $state )
```

### HybridSearch

```php
->search( $collection, $vector, $text, $limit = 5, $options = [] )
->searchAcross( $collections, $vector, $text, $limit = 5, $options = [] )
```

Options: `fetchK`, `vectorWeight`, `textWeight`, `rrfK`, `dimSlice`.

### IVFIndex

```php
new IVFIndex( StoreInterface $store, int $numClusters = 100, int $numProbes = 10 )

->build( $collection, $sampleDims = 128 ): array
->search( $collection, $query, $limit = 5, $dimSlice = 0 )
->matryoshkaSearch( $collection, $query, $limit, $stages, $multiplier = 3 )
->hasIndex( $collection ): bool
->indexStats( $collection ): ?array
->dropIndex( $collection )
```

### Math (static)

```php
VectorStore::normalize( $vector ): array
VectorStore::cosineSim( $a, $b, $dims ): float
VectorStore::euclideanDist( $a, $b, $dims ): float
VectorStore::dotProduct( $a, $b, $dims ): float
VectorStore::manhattanDist( $a, $b, $dims ): float
VectorStore::computeScore( $a, $b, $dims, Distance $distance ): float
```

## Storage Format

```
vectors/
├── articles.bin          ← Float32: N x dim x 4 bytes
├── articles.json         ← Manifest: IDs + metadata
├── articles.q8.bin       ← Int8: N x (dim + 8) bytes
├── articles.q8.json      ← Quantized manifest
├── articles.ivf.json     ← IVF: centroids + cluster assignments
├── articles.bm25.bin     ← BM25: inverted index (serialized PHP)
└── .htaccess             ← Access protection
```

## Testing

```bash
composer install
vendor/bin/phpunit
```

41 tests across 5 suites: VectorStore, QuantizedStore, IVFIndex, BM25, HybridSearch.

## Performance

### Speed (5,000 random vectors, PHP 8.2)

| Method | Time/query | Speedup |
|--------|-----------|---------|
| Brute-force 768d | 796ms | 1x |
| Matryoshka 128->384->768 | 182ms | 4.4x |
| IVF | 100ms | 7.9x |
| IVF + Matryoshka | **54ms** | **14.7x** |

### Storage

| Format | Per vector | 10K | 100K |
|--------|-----------|-----|------|
| Float32 768d | 3,072 B | 30 MB | 300 MB |
| Float32 384d | 1,536 B | 15 MB | 150 MB |
| Int8 768d | 776 B | 7.6 MB | 76 MB |
| **Int8 384d** | **392 B** | **3.8 MB** | **38 MB** |

## Integration Patterns

### WordPress

```php
$store = new QuantizedStore( WP_CONTENT_DIR . '/vectors', 384 );
$bm25  = new BM25\Index();

add_action( 'wp_after_insert_post', function( $id, $post ) use ( $store, $bm25 ) {
    if ( 'publish' !== $post->post_status ) return;
    $text   = $post->post_title . ' ' . wp_strip_all_tags( $post->post_content );
    $vector = array_slice( your_embedding_api( $text ), 0, 384 );
    $store->set( 'posts', (string) $id, $vector, ['title' => $post->post_title] );
    $bm25->addDocument( 'posts', (string) $id, $text );
    $store->flush();
    $bm25->save( WP_CONTENT_DIR . '/vectors', 'posts' );
}, 10, 2 );

// Hybrid search
$hybrid = new HybridSearch( $store, $bm25, HybridMode::RRF );
$results = $hybrid->search( 'posts', $query_vector, $search_text, 5 );
```

### Laravel

```php
// Service Provider
$this->app->singleton( StoreInterface::class, fn() =>
    new QuantizedStore( storage_path( 'vectors' ), 384 )
);

// Controller
public function search( Request $request ) {
    $store   = app( StoreInterface::class );
    $query   = array_slice( $this->embed( $request->q ), 0, 384 );
    $results = $store->matryoshkaSearch( 'articles', $query, 10, [128, 256, 384] );
    return Article::whereIn( 'id', array_column( $results, 'id' ) )->get();
}
```

### Neuron AI (RAG)

```php
use PHPVectorStore\Integration\NeuronVectorStore;

class MyRAG extends RAG {
    protected function vectorStore(): VectorStoreInterface {
        return new NeuronVectorStore(
            directory:  __DIR__ . '/vectors',
            dimensions: 384,
            quantized:  true,
            matryoshka: true,
        );
    }
}
```

## Architecture

```
PHPVectorStore\
├── StoreInterface           ← Common interface
├── VectorStore              ← Float32 storage (implements StoreInterface)
├── QuantizedStore           ← Int8 storage (implements StoreInterface)
├── IVFIndex                 ← K-means clustering (wraps StoreInterface)
├── HybridSearch             ← Vector + BM25 fusion
├── Distance                 ← Enum: Cosine, Euclidean, DotProduct, Manhattan
├── HybridMode               ← Enum: RRF, Weighted
├── Document                 ← Typed model
├── SearchResult             ← Typed model
├── BM25\
│   ├── Index                ← Okapi BM25 inverted index
│   ├── Config               ← k1, b parameters
│   ├── TokenizerInterface   ← Pluggable tokenization
│   └── SimpleTokenizer      ← Unicode tokenizer with stop words
├── Exception\
│   ├── VectorStoreException
│   ├── DimensionMismatchException
│   └── CollectionNotFoundException
└── Integration\
    └── NeuronVectorStore    ← Neuron AI RAG adapter
```

## License

GPL-2.0-or-later
