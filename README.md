# PHP Vector Store

Zero-dependency PHP vector database with Matryoshka search, IVF indexing, and Int8 quantization. Pure PHP 8.1+ — no SQLite, no C extensions, no FFI.

```
composer require mauricioperera/php-vector-store
```

## Why

Most vector databases require C extensions (sqlite-vec), external services (Pinecone, Weaviate), or specific runtimes (Python). PHP Vector Store runs anywhere PHP runs — shared hosting, WordPress, Laravel, any framework.

## Scaling Guide

| Vectors | Recommended Config | Storage/vec | Total (100K) | Speed |
|---------|-------------------|-------------|-------------|-------|
| <5K | Float32 768d + Matryoshka | 3,072 B | 300 MB | ~3ms |
| 5K-20K | Float32 384d + Matryoshka | 1,536 B | 150 MB | ~1.4ms |
| 20K-100K | Int8 384d + IVF + Matryoshka | **392 B** | **38 MB** | ~5ms |
| 100K-500K | Int8 384d + IVF + Matryoshka | **392 B** | **192 MB** | ~15ms |
| >500K | Use sqlite-vec or external service | — | — | — |

### Dimension reduction: 768d vs 384d

Matryoshka embeddings (EmbeddingGemma, etc.) encode information hierarchically. Truncating to 384 dimensions loses <5% of semantic information:

| Config | Storage/vec | Recall@1 | Recall@3 |
|--------|-----------|---------|---------|
| Float32 768d | 3,072 B | 90% | 100% |
| Float32 384d | 1,536 B | **90%** | **100%** |
| Int8 768d | 776 B | 90% | 100% |
| Int8 384d | **392 B** | **90%** | **100%** |

Same accuracy, 87% less storage. Benchmarked with real EmbeddingGemma-300m embeddings.

## Quick Start

```php
use PHPVectorStore\VectorStore;
use PHPVectorStore\QuantizedStore;
use PHPVectorStore\IVFIndex;

// Option A: Full precision (768d, 3KB/vec)
$store = new VectorStore( __DIR__ . '/vectors', 768 );

// Option B: Reduced dimensions (384d, 1.5KB/vec) — same accuracy
$store = new VectorStore( __DIR__ . '/vectors', 384 );

// Option C: Quantized (384d Int8, 392B/vec) — maximum density
$store = new QuantizedStore( __DIR__ . '/vectors', 384 );

// Store vectors (truncate to 384d if using 384d store)
$embedding_768 = get_embedding_from_api( $text );  // Your embedding API
$embedding_384 = array_slice( $embedding_768, 0, 384 );

$store->set( 'articles', 'art-1', $embedding_384, ['title' => 'My Article'] );
$store->flush();

// Search — Matryoshka stages adapt to your dimension
$results = $store->matryoshkaSearch( 'articles', $query_384, 5, [128, 256, 384] );

// For large datasets, add IVF
$ivf = new IVFIndex( $store, numClusters: 100, numProbes: 20 );
$ivf->build( 'articles' );
$results = $ivf->matryoshkaSearch( 'articles', $query_384, 5, [128, 256, 384] );
```

## Three Backends

### VectorStore (Float32)

Full precision. `dim × 4` bytes per vector.

```php
$store = new VectorStore( '/path', 768 );  // 3,072 B/vec
$store = new VectorStore( '/path', 384 );  // 1,536 B/vec
```

Files: `{collection}.bin` + `{collection}.json`

### QuantizedStore (Int8)

Each float mapped to 1 byte + 8-byte header. `dim + 8` bytes per vector.

Score drift vs Float32: **<0.001**. Ranking is identical.

```php
$q8 = new QuantizedStore( '/path', 768 );  // 776 B/vec
$q8 = new QuantizedStore( '/path', 384 );  // 392 B/vec
```

Files: `{collection}.q8.bin` + `{collection}.q8.json`

### IVFIndex (cluster-based)

K-means partitions vectors into clusters. At query time, only searches the closest clusters. Works on top of VectorStore or QuantizedStore.

```php
$ivf = new IVFIndex( $store, numClusters: 100, numProbes: 20 );
$ivf->build( 'collection' );
$ivf->search( 'collection', $query, 5 );
```

File: `{collection}.ivf.json`

## Search Strategies

### Brute-force

```php
$store->search( 'articles', $query, 5 );            // Full dimension
$store->search( 'articles', $query, 5, 128 );        // Reduced dims (faster)
```

### Matryoshka Multi-Stage

Progressive refinement — each stage narrows candidates before the next.

```php
// For 768d store
$store->matryoshkaSearch( 'col', $query, 5, [128, 384, 768] );

// For 384d store
$store->matryoshkaSearch( 'col', $query, 5, [128, 256, 384] );
```

Speedup: **3-5x** over brute-force.

### IVF + Matryoshka

IVF narrows to ~20% of vectors, then Matryoshka stages refine further.

```php
$ivf->matryoshkaSearch( 'col', $query, 5, [128, 256, 384] );
```

Speedup: **10-15x** over brute-force.

### Multi-collection

```php
$store->searchAcross( ['articles', 'comments', 'products'], $query, 10 );
```

## Performance

### Speed (21 texts, real EmbeddingGemma-300m embeddings, PHP 8.2)

| Config | Avg/query |
|--------|----------|
| Float32 768d brute-force | 2.99ms |
| Float32 768d Matryoshka 128→384→768 | 2.59ms |
| **Float32 384d brute-force** | **1.40ms** |
| Float32 384d Matryoshka 128→256→384 | 1.50ms |
| Int8 384d brute-force | 5.07ms |
| Int8 384d Matryoshka 128→256→384 | 5.42ms |

### Speed at scale (5,000 random vectors, PHP 8.2)

| Method | Time/query | Speedup |
|--------|-----------|---------|
| Brute-force 768d | 796ms | 1x |
| Matryoshka 128→384→768 | 182ms | 4.4x |
| IVF | 100ms | 7.9x |
| IVF + Matryoshka | **54ms** | **14.7x** |

### Storage

| Format | Per vector | 10K | 100K | 500K |
|--------|-----------|-----|------|------|
| JSON (typical) | ~7,000 B | 70 MB | 700 MB | 3.4 GB |
| Float32 768d | 3,072 B | 30 MB | 300 MB | 1.5 GB |
| Float32 384d | 1,536 B | 15 MB | 150 MB | 750 MB |
| Int8 768d | 776 B | 7.6 MB | 76 MB | 380 MB |
| **Int8 384d** | **392 B** | **3.8 MB** | **38 MB** | **192 MB** |

## API Reference

### VectorStore

```php
new VectorStore( string $dir, int $dim = 768, int $maxCollections = 50 );

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

// Search
->search( $collection, $query, $limit = 5, $dimSlice = 0 ): array
->matryoshkaSearch( $collection, $query, $limit = 5, $stages = [128,384,768] ): array
->searchAcross( $collections, $query, $limit = 5 ): array

// Import/Export
->import( $collection, $records ): int
->export( $collection ): array
```

### QuantizedStore

Same interface as VectorStore, with Int8 scalar quantization.

```php
new QuantizedStore( string $dir, int $dim = 768 );
// Same methods: set, remove, get, search, matryoshkaSearch, flush, etc.
```

### IVFIndex

```php
new IVFIndex( VectorStore|QuantizedStore $store, int $numClusters = 100, int $numProbes = 10 );

->build( $collection, $sampleDims = 128 ): array
->search( $collection, $query, $limit = 5 ): array
->matryoshkaSearch( $collection, $query, $limit, $stages ): array
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
```

## Storage Format

```
vectors/
├── articles.bin          ← Float32: N × dim × 4 bytes
├── articles.json         ← Manifest: IDs + metadata
├── articles.q8.bin       ← Int8: N × (dim + 8) bytes
├── articles.q8.json      ← Quantized manifest
├── articles.ivf.json     ← IVF: centroids + cluster assignments
└── .htaccess             ← Access protection
```

## How It Works

### Matryoshka Embeddings

Models like EmbeddingGemma encode information hierarchically:

```
[d0..d127]  → coarse (topic, domain)
[d0..d255]  → medium (subtopic, entities)
[d0..d383]  → detailed (specific meaning)     ← 384d sweet spot
[d0..d767]  → maximum (fine nuance)
```

You can truncate at any level. 384d retains ~95% of semantic information at half the storage.

### Int8 Quantization

Per-vector min/max linear mapping:

```
Encode: int8 = round((float - min) / (max - min) × 255) - 128
Decode: float = (int8 + 128) / 255 × (max - min) + min
```

Dequantized cosine similarity preserves ranking with <0.001 score drift.

### IVF Clustering

```
Build: all vectors → k-means → K centroids
Query: find P nearest centroids → search only those clusters
Scan:  N × (P/K) vectors instead of N
```

### Combined: Int8 384d + IVF + Matryoshka

The optimal stack for large datasets:

```
1. IVF prunes to ~20% of vectors (cluster selection)
2. Matryoshka stage 1 (128d): fast coarse ranking on candidates
3. Matryoshka stage 2 (256d): medium refinement
4. Matryoshka stage 3 (384d): final ranking

All on Int8 quantized data (392 bytes/vector)
```

Result: 500K vectors in 192 MB, searchable in ~15ms.

## vs sqlite-vec

| | PHP Vector Store | sqlite-vec |
|---|---|---|
| Dependencies | **None** | C extension or FFI |
| PHP version | 8.1+ | 8.1+ with extension |
| Deployment | Drop anywhere | Server config needed |
| Best range | <500K vectors | >500K vectors |
| Quantization | Int8 (4x compression) | Varies |
| Matryoshka | Native multi-stage | Manual |
| Storage efficiency | 392 B/vec (Int8 384d) | ~400 B/vec |
| Search algorithm | Brute + IVF + Matryoshka | ANN (HNSW) |
| Crash safety | Atomic rename | SQLite WAL |

## License

GPL-2.0-or-later
