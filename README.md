# PHP Vector Store

Zero-dependency PHP vector database with Matryoshka search, IVF indexing, and Int8 quantization. Pure PHP 8.1+ — no SQLite, no C extensions, no FFI.

```
composer require mauricioperera/php-vector-store
```

## When to use this

| | PHP Vector Store | sqlite-vec |
|---|---|---|
| Dependencies | **None** (pure PHP 8.1+) | C extension or FFI |
| Best for | <200K vectors | >200K vectors |
| Search | Brute-force + Matryoshka + IVF | ANN (IVF/HNSW) |
| Quantization | Int8 (4x compression, <0.001 drift) | Varies |
| Matryoshka | Native multi-stage (128→384→768) | Manual |
| Deployment | Drop-in anywhere | Requires extension install |

## Quick Start

```php
use PHPVectorStore\VectorStore;
use PHPVectorStore\QuantizedStore;
use PHPVectorStore\IVFIndex;

// Float32 store (3072 bytes/vector)
$store = new VectorStore( __DIR__ . '/vectors', 768 );

// Int8 quantized store (776 bytes/vector — 4x smaller, same accuracy)
$q8 = new QuantizedStore( __DIR__ . '/vectors', 768 );

// Store a vector
$store->set( 'articles', 'art-1', $embedding, ['title' => 'My Article'] );

// Search
$results = $store->search( 'articles', $queryVector, 5 );

// Matryoshka search (3-5x faster)
$results = $store->matryoshkaSearch( 'articles', $queryVector, 5 );

// IVF index for large datasets (10-15x faster)
$ivf = new IVFIndex( $store, numClusters: 100, numProbes: 20 );
$ivf->build( 'articles' );
$results = $ivf->matryoshkaSearch( 'articles', $queryVector, 5 );

// Persist to disk
$store->flush();
```

## Scaling Strategy

| Vectors | Recommended | Storage/vec | Search speed |
|---------|------------|-------------|-------------|
| <5K | VectorStore + Matryoshka | 3,072 B | ~3ms |
| 5K-20K | QuantizedStore + Matryoshka | 776 B | ~10ms |
| 20K-100K | VectorStore + IVF + Matryoshka | 3,072 B | ~50ms |
| 100K-200K | QuantizedStore + IVF + Matryoshka | 776 B | ~50ms |
| >200K | Use sqlite-vec or external service | — | — |

## Three Storage Backends

### VectorStore (Float32)

Full precision. 768 × 4 bytes = **3,072 bytes per vector**.

```php
$store = new VectorStore( '/path/to/dir', 768 );
$store->set( 'collection', 'id', $vector, $metadata );
$store->search( 'collection', $query, 5 );
```

Files: `collection.bin` (raw Float32) + `collection.json` (manifest).

### QuantizedStore (Int8)

Each float quantized to 1 byte + 8-byte header (min/max). **776 bytes per vector** — 75% smaller.

Score drift vs Float32: **<0.001**. Ranking is identical.

```php
$q8 = new QuantizedStore( '/path/to/dir', 768 );
$q8->set( 'collection', 'id', $vector, $metadata );
$q8->search( 'collection', $query, 5 );
```

Files: `collection.q8.bin` + `collection.q8.json`.

### IVFIndex (cluster-based)

Partitions vectors into K clusters via k-means. At query time, only searches the P closest clusters. Works on top of VectorStore or QuantizedStore.

```php
$ivf = new IVFIndex( $store, numClusters: 100, numProbes: 20 );
$ivf->build( 'collection', sampleDims: 128 );  // One-time k-means
$ivf->search( 'collection', $query, 5 );
```

File: `collection.ivf.json` (centroids + assignments).

## Search Strategies

### 1. Brute-force

Compares query against every vector. Exact results, O(N).

```php
$store->search( 'articles', $query, 5 );
```

### 2. Matryoshka Multi-Stage (128→384→768)

Exploits hierarchical structure in Matryoshka embeddings. Three passes, each narrowing candidates before the next more expensive comparison.

```php
$store->matryoshkaSearch( 'articles', $query, 5, [128, 384, 768] );
```

Speedup: **3-5x**. Recall: **~100%**.

### 3. IVF (Inverted File Index)

K-means clustering partitions vectors into groups. Only searches the closest clusters.

```php
$ivf = new IVFIndex( $store, numClusters: 100, numProbes: 20 );
$ivf->build( 'articles' );
$ivf->search( 'articles', $query, 5 );
```

Speedup: **5-8x**. Recall: **80-95%** (higher with real semantic data).

### 4. IVF + Matryoshka (fastest)

IVF narrows to ~20% of vectors, then Matryoshka stages refine further.

```php
$ivf->matryoshkaSearch( 'articles', $query, 5, [128, 384, 768] );
```

Speedup: **10-15x**. Best strategy for 5K-200K vectors.

### 5. Multi-collection search

Search across multiple collections in one call.

```php
$store->searchAcross( ['articles', 'comments', 'products'], $query, 10 );
```

## Real-World Benchmark

Tested with **EmbeddingGemma-300m** (768d) via Cloudflare Workers AI on 42 texts across 9 topics:

### Accuracy (10 semantic queries)

| Method | Recall@1 | Recall@3 |
|--------|---------|---------|
| Float32 brute-force | 90% | 100% |
| Float32 Matryoshka | 90% | 100% |
| Int8 brute-force | **90%** | **100%** |
| Int8 Matryoshka | **90%** | **100%** |

Int8 quantization preserves ranking perfectly. Score drift: **<0.001**.

### Storage comparison

| Format | Per vector | 10K vectors | 100K vectors |
|--------|-----------|-------------|-------------|
| JSON (typical) | ~7,000 B | 70 MB | 700 MB |
| Float32 binary | 3,072 B | 30 MB | 300 MB |
| **Int8 quantized** | **776 B** | **7.6 MB** | **76 MB** |

### Speed (5,000 vectors, PHP 8.2)

| Method | Time/query | Speedup |
|--------|-----------|---------|
| Brute-force 768d | 796ms | 1x |
| Matryoshka 128→384→768 | 182ms | 4.4x |
| IVF (K=71) | 100ms | 7.9x |
| IVF + Matryoshka | **54ms** | **14.7x** |

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
->searchAcross( $collections, $query, $limit = 5, $dimSlice = 0 ): array

// Import/Export
->import( $collection, $records ): int
->export( $collection ): array
```

### QuantizedStore

Same interface as VectorStore, with Int8 quantization.

```php
new QuantizedStore( string $dir, int $dim = 768 );

// Same methods: set, remove, get, has, count, search, matryoshkaSearch, flush, etc.

// Additional
QuantizedStore::quantize( $vector, $dim ): string   // Float→Int8
QuantizedStore::dequantize( $bin, $offset, $dim ): array  // Int8→Float
```

### IVFIndex

```php
new IVFIndex( VectorStore $store, int $numClusters = 100, int $numProbes = 10 );

->build( $collection, $sampleDims = 128 ): array  // K-means clustering
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
├── articles.bin         ← Float32: N × 768 × 4 bytes
├── articles.json        ← Manifest: IDs + metadata
├── articles.q8.bin      ← Int8: N × (768 + 8) bytes
├── articles.q8.json     ← Quantized manifest
├── articles.ivf.json    ← IVF index: centroids + clusters
└── .htaccess            ← Access protection (WordPress)
```

## How It Works

### Binary Storage

Vectors stored as contiguous raw bytes. No JSON, no SQL, no serialization overhead.

Float32: `[f32][f32][f32]...[f32]` × N vectors
Int8: `[min_f32][max_f32][i8][i8]...[i8]` × N vectors

### Matryoshka Embeddings

Models like EmbeddingGemma encode information hierarchically:

```
[d0..d127]  → coarse features (topic, domain)
[d0..d383]  → medium features (subtopic, entities)
[d0..d767]  → fine features (specific meaning, nuance)
```

Multi-stage search exploits this: cheap coarse pass eliminates 90% of candidates before the expensive fine pass.

### IVF Clustering

K-means partitions the vector space into K regions. Each query only searches the P nearest regions:

```
Build: vectors → k-means → K centroids + assignments
Query: find P nearest centroids → search only those clusters
Scan: N × (P/K) vectors instead of N
```

### Int8 Quantization

Maps each float to [-128, 127] using per-vector min/max scaling:

```
Encode: int8 = round((float - min) / (max - min) × 255) - 128
Decode: float = (int8 + 128) / 255 × (max - min) + min
```

Per-vector headers preserve scale, so dequantized cosine similarity matches Float32 with <0.001 drift.

## License

GPL-2.0-or-later
