# PHP Vector Store

Zero-dependency PHP vector database with Matryoshka search. Stores embeddings as raw Float32 binary files — no SQLite, no C extensions, no FFI.

A lightweight alternative to `sqlite-vec` for datasets under 50K vectors.

## When to use this vs sqlite-vec

| | PHP Vector Store | sqlite-vec |
|---|---|---|
| Dependencies | **None** (pure PHP) | C extension or FFI |
| Best for | <50K vectors | >50K vectors |
| Search | Brute-force + Matryoshka | ANN (IVF/HNSW) |
| Size per 768d vector | 3,072 bytes | ~3,100 bytes |
| Matryoshka support | Native | Manual |
| WordPress integration | Drop-in | Requires extension install |
| PHP requirement | 8.1+ | 8.1+ with FFI/extension |

**Use PHP Vector Store when:**
- You need vector search without installing C extensions
- Your dataset is under 50K vectors
- You want zero-config deployment (shared hosting, WordPress, etc.)
- You use Matryoshka embeddings (EmbeddingGemma, etc.)

**Use sqlite-vec when:**
- You have >50K vectors and need ANN indexing
- You can install C extensions on your server
- You need SQL-based querying alongside vectors

## Installation

Copy `src/VectorStore.php` to your project. No Composer needed.

```php
require_once 'path/to/VectorStore.php';

use PHPVectorStore\VectorStore;
$store = new VectorStore( '/path/to/storage', 768 );
```

## Quick Start

```php
use PHPVectorStore\VectorStore;

$store = new VectorStore( __DIR__ . '/vectors' );

// Store vectors
$store->set( 'articles', 'article-1', $embedding, ['title' => 'My Article'] );
$store->set( 'articles', 'article-2', $embedding2, ['title' => 'Another'] );

// Search (cosine similarity)
$results = $store->search( 'articles', $queryVector, 5 );
// [['id' => 'article-1', 'score' => 0.95, 'metadata' => ['title' => 'My Article']]]

// Matryoshka search (6x faster)
$results = $store->matryoshkaSearch( 'articles', $queryVector, 5, 128 );
// Coarse pass at 128d, fine re-rank at 768d

// Persist to disk
$store->flush();
```

## API

### Constructor

```php
new VectorStore(
    string $directory,       // Storage directory (created if needed)
    int    $dimensions = 768, // Vector dimensions
    int    $maxCollections = 50 // Max collections in memory (LRU)
);
```

### Write

```php
$store->set( $collection, $id, $vector, $metadata = [] );  // Insert/update
$store->remove( $collection, $id );                         // Delete
$store->drop( $collection );                                // Delete collection
$store->flush();                                            // Persist to disk
```

### Read

```php
$store->get( $collection, $id );        // Single vector + metadata
$store->has( $collection, $id );        // Exists check
$store->count( $collection );           // Vector count
$store->ids( $collection );             // All IDs
$store->collections();                  // All collection names
$store->stats();                        // Storage statistics
```

### Search

```php
// Full-resolution search (768d)
$store->search( $collection, $queryVector, $limit = 5 );

// Matryoshka search (coarse 128d → fine 768d)
$store->matryoshkaSearch( $collection, $queryVector, $limit = 5, $coarseDims = 128 );

// Reduced-dimension search (faster, slightly less accurate)
$store->search( $collection, $queryVector, $limit, $dimSlice = 128 );

// Multi-collection search
$store->searchAcross( ['articles', 'comments'], $queryVector, $limit );
```

### Import / Export

```php
// Import from JSON array
$store->import( 'articles', [
    ['id' => 'a1', 'vector' => [...], 'metadata' => ['title' => '...']],
]);

// Export to JSON-serializable array
$data = $store->export( 'articles' );
```

### Math utilities (static)

```php
VectorStore::normalize( $vector );              // L2 normalize
VectorStore::cosineSim( $a, $b, $dims );        // Cosine similarity
VectorStore::euclideanDist( $a, $b, $dims );    // Euclidean distance
VectorStore::dotProduct( $a, $b, $dims );       // Dot product
```

## Storage Format

```
vectors/
├── articles.bin   ← Raw Float32 (N × 768 × 4 bytes)
├── articles.json  ← Manifest: IDs, metadata, version
├── comments.bin
└── comments.json
```

Each `.bin` file is a contiguous array of Float32 values. No headers, no padding, no alignment tricks. Just `N × dim × 4` bytes of raw floats.

The `.json` manifest maps array positions to entity IDs and stores metadata:

```json
{
  "version": 1,
  "dim": 768,
  "count": 1000,
  "ids": ["article-1", "article-2", ...],
  "meta": {
    "article-1": {"title": "My Article"},
    "article-2": {"title": "Another"}
  }
}
```

## How Matryoshka Search Works

Matryoshka embeddings (like EmbeddingGemma) encode information hierarchically: the first 128 dimensions capture the most important features, 256 captures more, and 768 captures everything.

```
Full vector: [d0, d1, d2, ..., d127, d128, ..., d255, d256, ..., d767]
              ├── 128d (coarse) ──┤
              ├────── 256d (medium) ──────┤
              ├────────── 768d (fine) ──────────────────────────────┤
```

Matryoshka search exploits this:

1. **Coarse pass** (128d): Compare only first 128 dimensions. 6x less computation. Gets top N×3 candidates.
2. **Fine re-rank** (768d): Re-score only the candidates using all 768 dimensions.

Result: Nearly the same accuracy as full 768d search, but ~3-5x faster.

## Performance

Benchmarked on PHP 8.2, single thread:

| Vectors | Insert | Full Search (768d) | Matryoshka (128→768) | Storage |
|---------|--------|-------------------|---------------------|---------|
| 100 | 15ms | 0.3ms | 0.2ms | 300 KB |
| 1,000 | 120ms | 3ms | 1.5ms | 3 MB |
| 5,000 | 600ms | 15ms | 6ms | 15 MB |
| 10,000 | 1.2s | 30ms | 12ms | 30 MB |
| 50,000 | 6s | 150ms | 60ms | 150 MB |

For comparison, sqlite-vec with HNSW index on 50K vectors: ~5ms/query. But requires C extension installation.

## License

GPL-2.0-or-later
