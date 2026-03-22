# PHP Vector Store

Zero-dependency PHP vector database with Matryoshka search and IVF indexing. Stores embeddings as raw Float32 binary files — no SQLite, no C extensions, no FFI.

```
composer require mauricioperera/php-vector-store
```

## When to use this

| | PHP Vector Store | sqlite-vec |
|---|---|---|
| Dependencies | **None** (pure PHP 8.1+) | C extension or FFI |
| Search <5K vectors | Matryoshka brute-force | Overkill |
| Search 5K-100K vectors | **IVF + Matryoshka** | ANN (IVF/HNSW) |
| Search >100K vectors | Not recommended | Use this instead |
| Size per 768d vector | 3,072 bytes | ~3,100 bytes |
| Deployment | Drop-in anywhere | Requires extension install |

## Quick Start

```php
use PHPVectorStore\VectorStore;
use PHPVectorStore\IVFIndex;

$store = new VectorStore( __DIR__ . '/vectors', 768 );

// Store vectors
$store->set( 'articles', 'art-1', $embedding, ['title' => 'First Article'] );
$store->set( 'articles', 'art-2', $embedding2, ['title' => 'Second Article'] );
$store->flush();

// Search — brute-force (best for <5K vectors)
$results = $store->search( 'articles', $queryVector, 5 );

// Matryoshka search — 3-5x faster (best for <10K vectors)
$results = $store->matryoshkaSearch( 'articles', $queryVector, 5 );
// Default stages: 128d → 384d → 768d

// IVF + Matryoshka — 10-15x faster (best for 5K-100K vectors)
$ivf = new IVFIndex( $store, numClusters: 100, numProbes: 20 );
$ivf->build( 'articles' );  // K-means clustering (one-time)
$results = $ivf->matryoshkaSearch( 'articles', $queryVector, 5 );
```

## Search Strategies

### 1. Brute-force (default)

Compares query against every vector. Simple, exact, O(N).

```php
$store->search( $collection, $query, $limit );
```

Best for: <5K vectors.

### 2. Matryoshka Multi-Stage

Exploits Matryoshka embeddings (EmbeddingGemma, etc.) where the first N dimensions capture the most important features.

```php
$store->matryoshkaSearch( $collection, $query, $limit, [128, 384, 768] );
```

Three passes:
1. **128d** — scan all vectors (cheap, 6x less computation)
2. **384d** — re-rank top candidates
3. **768d** — final re-rank

Result: 3-5x speedup, ~100% recall. Best for: <10K vectors.

### 3. IVF (Inverted File Index)

Partitions vectors into K clusters via k-means. At query time, only searches the P closest clusters.

```php
$ivf = new IVFIndex( $store, numClusters: 100, numProbes: 20 );
$ivf->build( 'articles' );
$ivf->search( $collection, $query, $limit );
```

Scans only N×(P/K) vectors instead of N. Best for: 5K-100K vectors.

### 4. IVF + Matryoshka (fastest)

Combines IVF cluster pruning with Matryoshka multi-stage refinement.

```php
$ivf->matryoshkaSearch( $collection, $query, $limit, [128, 384, 768] );
```

IVF narrows to ~20% of vectors, then Matryoshka stages refine further. **10-15x speedup** over brute-force. Best for: 5K-100K vectors.

## Performance

| Vectors | Brute-force 768d | Matryoshka 3-stage | IVF | IVF + Matryoshka |
|---------|-----------------|-------------------|-----|-----------------|
| 1,000 | 134ms | 29ms (4.7x) | 28ms (4.8x) | **17ms (7.9x)** |
| 5,000 | 796ms | 182ms (4.4x) | 100ms (7.9x) | **54ms (14.7x)** |

Storage: 3,072 bytes per 768d vector (3 MB per 1,000 vectors).

## API Reference

### VectorStore

```php
new VectorStore( string $directory, int $dimensions = 768, int $maxCollections = 50 );

// Write
$store->set( $collection, $id, $vector, $metadata = [] );
$store->remove( $collection, $id );
$store->drop( $collection );
$store->flush();

// Read
$store->get( $collection, $id );       // → {id, vector, metadata} | null
$store->has( $collection, $id );       // → bool
$store->count( $collection );          // → int
$store->ids( $collection );            // → string[]
$store->collections();                 // → string[]
$store->stats();                       // → {dimensions, total_vectors, ...}

// Search
$store->search( $collection, $query, $limit, $dimSlice = 0 );
$store->matryoshkaSearch( $collection, $query, $limit, $stages = [128, 384, 768] );
$store->searchAcross( $collections, $query, $limit, $dimSlice = 0 );

// Import / Export
$store->import( $collection, $records );
$store->export( $collection );
```

### IVFIndex

```php
new IVFIndex( VectorStore $store, int $numClusters = 100, int $numProbes = 10 );

$ivf->build( $collection, $sampleDims = 128 );   // Build k-means index
$ivf->search( $collection, $query, $limit );      // IVF search
$ivf->matryoshkaSearch( $collection, $query, $limit, $stages ); // IVF + Matryoshka
$ivf->hasIndex( $collection );                     // Check if index exists
$ivf->indexStats( $collection );                   // Index statistics
$ivf->dropIndex( $collection );                    // Remove index
```

### Math Utilities (static)

```php
VectorStore::normalize( $vector );             // L2 normalize
VectorStore::cosineSim( $a, $b, $dims );       // Cosine similarity
VectorStore::euclideanDist( $a, $b, $dims );   // Euclidean distance
VectorStore::dotProduct( $a, $b, $dims );      // Dot product
```

## Storage Format

```
vectors/
├── articles.bin       ← Raw Float32 (N × 768 × 4 bytes)
├── articles.json      ← Manifest: IDs, metadata, version
├── articles.ivf.json  ← IVF index: centroids, cluster assignments
├── comments.bin
└── comments.json
```

## How IVF Works

```
Build (one-time):
  1. Run k-means on all vectors at 128d → K cluster centroids
  2. Assign each vector to its nearest centroid
  3. Save centroid positions + cluster memberships

Search:
  1. Find P closest centroids to query (P << K)
  2. Only compare vectors in those P clusters
  3. Reduction: scan N×(P/K) vectors instead of N

Tuning:
  K (clusters) = sqrt(N) is a good default
  P (probes) = 10-20% of K balances speed vs recall
  More probes = better recall, slower search
```

## License

GPL-2.0-or-later
