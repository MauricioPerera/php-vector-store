<?php
/**
 * PHP Vector Store — Binary Float32 vector storage with Matryoshka search.
 *
 * A zero-dependency PHP vector database that stores embeddings as raw Float32
 * binary files. Designed as a lightweight alternative to sqlite-vec for
 * small-to-medium datasets (<100K vectors).
 *
 * Features:
 * - Binary storage: 768d vector = 3072 bytes (vs ~7KB JSON, vs ~6KB SQLite BLOB)
 * - Matryoshka search: coarse pass at 128d, fine re-rank at 768d (6x speedup)
 * - Per-collection scoping with LRU eviction
 * - No external dependencies (no SQLite extension, no FFI, no C libraries)
 * - Atomic writes for crash safety
 * - Metadata support per vector (JSON sidecar)
 *
 * Storage format:
 *   {dir}/{collection}.bin  — contiguous Float32 array (N vectors × DIM floats)
 *   {dir}/{collection}.json — manifest: IDs, metadata, dimension, version
 *
 * @package PHPVectorStore
 * @version 0.1.0
 * @license GPL-2.0-or-later
 */

namespace PHPVectorStore;

class VectorStore implements StoreInterface {

	private const FORMAT_VERSION = 1;
	private const BYTES_PER_F32  = 4;

	private int    $dim;
	private string $dir;
	private int    $maxCollections;

	/** @var array<string, array{ids: string[], id_map: array<string,int>, bin: string, meta: array<string,array>}> */
	private array $cache = array();
	private array $dirty = array();

	/**
	 * @param string $directory       Storage directory (will be created if needed).
	 * @param int    $dimensions      Vector dimensions (default 768 for EmbeddingGemma).
	 * @param int    $maxCollections  Max collections kept in memory (LRU eviction).
	 */
	public function __construct( string $directory, int $dimensions = 768, int $maxCollections = 50 ) {
		$this->dir            = rtrim( $directory, '/\\' );
		$this->dim            = $dimensions;
		$this->maxCollections = $maxCollections;

		if ( ! is_dir( $this->dir ) ) {
			mkdir( $this->dir, 0755, true );
		}
	}

	public function dimensions(): int {
		return $this->dim;
	}

	public function directory(): string {
		return $this->dir;
	}

	// ── Write Operations ───────────────────────────────────────────────

	/**
	 * Store or overwrite a vector.
	 *
	 * @param string $collection Collection name (e.g., 'posts', 'memories').
	 * @param string $id         Unique identifier for this vector.
	 * @param float[] $vector    Vector of $dim floats (will be L2-normalized).
	 * @param array  $metadata   Optional metadata to store alongside the vector.
	 */
	public function set( string $collection, string $id, array $vector, array $metadata = array() ): void {
		if ( count( $vector ) < $this->dim ) {
			throw Exception\DimensionMismatchException::forVectors( $this->dim, count( $vector ) );
		}

		$col = $this->loadCollection( $collection );

		$normalized = self::normalize( $vector );
		$binary     = self::packVector( $normalized, $this->dim );

		$pos = $col['id_map'][ $id ] ?? null;

		if ( null !== $pos ) {
			// Overwrite in place
			$offset      = $pos * $this->bytesPerVector();
			$col['bin']  = substr_replace( $col['bin'], $binary, $offset, $this->bytesPerVector() );
			$col['meta'][ $id ] = $metadata;
		} else {
			// Append
			$col['ids'][]          = $id;
			$col['id_map'][ $id ]  = count( $col['ids'] ) - 1;
			$col['bin']           .= $binary;
			$col['meta'][ $id ]    = $metadata;
		}

		$this->cache[ $collection ] = $col;
		$this->dirty[ $collection ] = true;
	}

	/**
	 * Remove a vector. Returns true if it existed.
	 */
	public function remove( string $collection, string $id ): bool {
		$col = $this->loadCollection( $collection );
		$pos = $col['id_map'][ $id ] ?? null;

		if ( null === $pos ) {
			return false;
		}

		$bpv      = $this->bytesPerVector();
		$last_idx = count( $col['ids'] ) - 1;

		if ( $pos !== $last_idx ) {
			// Swap with last
			$last_id                    = $col['ids'][ $last_idx ];
			$col['ids'][ $pos ]         = $last_id;
			$col['id_map'][ $last_id ]  = $pos;

			$src = substr( $col['bin'], $last_idx * $bpv, $bpv );
			$col['bin'] = substr_replace( $col['bin'], $src, $pos * $bpv, $bpv );
		}

		array_pop( $col['ids'] );
		unset( $col['id_map'][ $id ], $col['meta'][ $id ] );
		$col['bin'] = substr( $col['bin'], 0, count( $col['ids'] ) * $bpv );

		$this->cache[ $collection ] = $col;
		$this->dirty[ $collection ] = true;
		return true;
	}

	// ── Read Operations ────────────────────────────────────────────────

	/**
	 * Get a single vector with its metadata.
	 *
	 * @return array{id: string, vector: float[], metadata: array}|null
	 */
	public function get( string $collection, string $id ): ?array {
		$col = $this->loadCollection( $collection );
		$pos = $col['id_map'][ $id ] ?? null;

		if ( null === $pos ) {
			return null;
		}

		$offset = $pos * $this->bytesPerVector();
		$floats = array_values( unpack( 'f' . $this->dim, $col['bin'], $offset ) );

		return array(
			'id'       => $id,
			'vector'   => $floats,
			'metadata' => $col['meta'][ $id ] ?? array(),
		);
	}

	/**
	 * Check if a vector exists.
	 */
	public function has( string $collection, string $id ): bool {
		$col = $this->loadCollection( $collection );
		return isset( $col['id_map'][ $id ] );
	}

	/**
	 * Count vectors in a collection.
	 */
	public function count( string $collection ): int {
		$col = $this->loadCollection( $collection );
		return count( $col['ids'] );
	}

	/**
	 * List all IDs in a collection.
	 */
	public function ids( string $collection ): array {
		$col = $this->loadCollection( $collection );
		return $col['ids'];
	}

	/**
	 * List all collections.
	 */
	public function collections(): array {
		$files = glob( $this->dir . '/*.json' );
		return array_map( fn( $f ) => basename( $f, '.json' ), $files ?: array() );
	}

	// ── Search ─────────────────────────────────────────────────────────

	/**
	 * Cosine similarity search.
	 *
	 * @param string  $collection  Collection to search.
	 * @param float[] $query       Query vector (will be normalized).
	 * @param int     $limit       Max results.
	 * @param int     $dimSlice    Dimensions to compare (Matryoshka: 128, 256, or full).
	 * @return array<array{id: string, score: float, metadata: array}>
	 */
	public function search( string $collection, array $query, int $limit = 5, int $dimSlice = 0, ?Distance $distance = null ): array {
		$col = $this->loadCollection( $collection );

		if ( empty( $col['ids'] ) ) {
			return array();
		}

		$distance = $distance ?? Distance::Cosine;
		$dims     = $dimSlice > 0 ? min( $dimSlice, $this->dim ) : $this->dim;
		$q        = self::normalize( $query );
		$count    = count( $col['ids'] );
		$bpv      = $this->bytesPerVector();

		$results = array();

		for ( $i = 0; $i < $count; $i++ ) {
			$v     = array_values( unpack( 'f' . $dims, $col['bin'], $i * $bpv ) );
			$score = self::computeScore( $q, $v, $dims, $distance );

			if ( $score > 0 ) {
				$results[] = array(
					'id'       => $col['ids'][ $i ],
					'score'    => round( $score, 6 ),
					'metadata' => $col['meta'][ $col['ids'][ $i ] ] ?? array(),
				);
			}
		}

		usort( $results, fn( $a, $b ) => $b['score'] <=> $a['score'] );
		return array_slice( $results, 0, $limit );
	}

	/**
	 * Matryoshka search: multi-stage progressive refinement.
	 *
	 * Default 3 stages: 128d → 384d → 768d
	 * Each stage narrows the candidate set before comparing at higher resolution.
	 *
	 * @param string  $collection          Collection to search.
	 * @param float[] $query               Query vector.
	 * @param int     $limit               Final results.
	 * @param int[]   $stages              Dimension stages (default [128, 384, 768]).
	 * @param int     $candidateMultiplier Each stage keeps limit × multiplier candidates.
	 * @return array<array{id: string, score: float, stages: array, metadata: array}>
	 */
	public function matryoshkaSearch(
		string $collection,
		array  $query,
		int    $limit = 5,
		array  $stages = array( 128, 384, 768 ),
		int    $candidateMultiplier = 3,
		?Distance $distance = null
	): array {
		$col = $this->loadCollection( $collection );

		if ( empty( $col['ids'] ) ) {
			return array();
		}

		$distance = $distance ?? Distance::Cosine;
		$q        = self::normalize( $query );
		$bpv      = $this->bytesPerVector();
		$count    = count( $col['ids'] );

		// Ensure stages are sorted and capped at dim
		sort( $stages );
		$stages = array_map( fn( $s ) => min( $s, $this->dim ), $stages );

		// Start with all indices as candidates
		$candidate_indices = range( 0, $count - 1 );
		$stage_scores      = array(); // id => [stage_dim => score]

		foreach ( $stages as $si => $dims ) {
			$is_last = $si === count( $stages ) - 1;
			$keep    = $is_last ? $limit : $limit * $candidateMultiplier;

			$scored = array();

			foreach ( $candidate_indices as $i ) {
				$v     = array_values( unpack( 'f' . $dims, $col['bin'], $i * $bpv ) );
				$score = self::computeScore( $q, $v, $dims, $distance );

				$scored[] = array( 'index' => $i, 'score' => $score );

				$id = $col['ids'][ $i ];
				if ( ! isset( $stage_scores[ $id ] ) ) {
					$stage_scores[ $id ] = array();
				}
				$stage_scores[ $id ][ $dims ] = round( $score, 6 );
			}

			// Sort and keep top candidates for next stage
			usort( $scored, fn( $a, $b ) => $b['score'] <=> $a['score'] );
			$scored            = array_slice( $scored, 0, $keep );
			$candidate_indices = array_map( fn( $s ) => $s['index'], $scored );
		}

		// Build final results from last stage's survivors
		$last_dim = end( $stages );
		$results  = array();

		foreach ( $candidate_indices as $i ) {
			$id = $col['ids'][ $i ];
			$results[] = array(
				'id'       => $id,
				'score'    => $stage_scores[ $id ][ $last_dim ] ?? 0.0,
				'stages'   => $stage_scores[ $id ] ?? array(),
				'metadata' => $col['meta'][ $id ] ?? array(),
			);
		}

		usort( $results, fn( $a, $b ) => $b['score'] <=> $a['score'] );
		return array_slice( $results, 0, $limit );
	}

	/**
	 * Multi-collection search. Merges results, keeping max score per ID.
	 */
	public function searchAcross( array $collections, array $query, int $limit = 5, int $dimSlice = 0, ?Distance $distance = null ): array {
		$merged = array();

		foreach ( $collections as $col ) {
			foreach ( $this->search( $col, $query, $limit, $dimSlice, $distance ) as $r ) {
				$key = $col . ':' . $r['id'];
				if ( ! isset( $merged[ $key ] ) || $r['score'] > $merged[ $key ]['score'] ) {
					$r['collection'] = $col;
					$merged[ $key ]  = $r;
				}
			}
		}

		$all = array_values( $merged );
		usort( $all, fn( $a, $b ) => $b['score'] <=> $a['score'] );
		return array_slice( $all, 0, $limit );
	}

	// ── Persistence ────────────────────────────────────────────────────

	/**
	 * Flush all dirty collections to disk.
	 */
	public function flush(): void {
		foreach ( $this->dirty as $collection => $_ ) {
			$this->persistCollection( $collection );
		}
		$this->dirty = array();
	}

	/**
	 * Drop a collection (delete files + clear cache).
	 */
	public function drop( string $collection ): void {
		$json = $this->dir . '/' . $collection . '.json';
		$bin  = $this->dir . '/' . $collection . '.bin';

		if ( file_exists( $json ) ) unlink( $json );
		if ( file_exists( $bin ) )  unlink( $bin );

		unset( $this->cache[ $collection ], $this->dirty[ $collection ] );
	}

	/**
	 * Get storage statistics.
	 */
	public function stats(): array {
		$total_vectors = 0;
		$total_bytes   = 0;
		$detail        = array();

		foreach ( $this->collections() as $name ) {
			$bin_path = $this->dir . '/' . $name . '.bin';
			$size     = file_exists( $bin_path ) ? filesize( $bin_path ) : 0;
			$vectors  = (int) ( $size / $this->bytesPerVector() );

			$total_vectors += $vectors;
			$total_bytes   += $size;
			$detail[]       = array( 'collection' => $name, 'vectors' => $vectors, 'bytes' => $size );
		}

		return array(
			'dimensions'    => $this->dim,
			'total_vectors' => $total_vectors,
			'total_bytes'   => $total_bytes,
			'memory_mb'     => round( $total_bytes / ( 1024 * 1024 ), 3 ),
			'bytes_per_vec' => $this->bytesPerVector(),
			'collections'   => $detail,
		);
	}

	// ── Import / Export ────────────────────────────────────────────────

	/**
	 * Import vectors from a JSON array.
	 *
	 * @param string $collection Target collection.
	 * @param array  $records    Array of ['id' => string, 'vector' => float[], 'metadata' => array].
	 * @return int Number of vectors imported.
	 */
	public function import( string $collection, array $records ): int {
		$count = 0;
		foreach ( $records as $r ) {
			if ( empty( $r['id'] ) || empty( $r['vector'] ) ) continue;
			$this->set( $collection, $r['id'], $r['vector'], $r['metadata'] ?? array() );
			$count++;
		}
		return $count;
	}

	/**
	 * Export all vectors from a collection as JSON-serializable array.
	 */
	public function export( string $collection ): array {
		$col    = $this->loadCollection( $collection );
		$bpv    = $this->bytesPerVector();
		$result = array();

		foreach ( $col['ids'] as $i => $id ) {
			$floats   = array_values( unpack( 'f' . $this->dim, $col['bin'], $i * $bpv ) );
			$result[] = array(
				'id'       => $id,
				'vector'   => $floats,
				'metadata' => $col['meta'][ $id ] ?? array(),
			);
		}

		return $result;
	}

	// ── Private ────────────────────────────────────────────────────────

	private function bytesPerVector(): int {
		return $this->dim * self::BYTES_PER_F32;
	}

	private function loadCollection( string $name ): array {
		if ( isset( $this->cache[ $name ] ) ) {
			return $this->cache[ $name ];
		}

		$json_path = $this->dir . '/' . $name . '.json';
		$bin_path  = $this->dir . '/' . $name . '.bin';

		if ( ! file_exists( $json_path ) || ! file_exists( $bin_path ) ) {
			$col = array( 'ids' => array(), 'id_map' => array(), 'bin' => '', 'meta' => array() );
			$this->cache[ $name ] = $col;
			$this->evictLru();
			return $col;
		}

		$manifest = json_decode( file_get_contents( $json_path ), true );
		if ( ! $manifest || ( $manifest['version'] ?? 0 ) !== self::FORMAT_VERSION ) {
			$col = array( 'ids' => array(), 'id_map' => array(), 'bin' => '', 'meta' => array() );
			$this->cache[ $name ] = $col;
			return $col;
		}

		$bin    = file_get_contents( $bin_path );
		$ids    = $manifest['ids'];
		$id_map = array();
		foreach ( $ids as $i => $id ) {
			$id_map[ $id ] = $i;
		}

		$col = array(
			'ids'    => $ids,
			'id_map' => $id_map,
			'bin'    => $bin,
			'meta'   => $manifest['meta'] ?? array(),
		);

		$this->cache[ $name ] = $col;
		$this->evictLru();
		return $col;
	}

	private function persistCollection( string $name ): void {
		if ( ! isset( $this->cache[ $name ] ) ) return;

		$col = $this->cache[ $name ];

		if ( empty( $col['ids'] ) ) {
			@unlink( $this->dir . '/' . $name . '.json' );
			@unlink( $this->dir . '/' . $name . '.bin' );
			return;
		}

		$manifest = array(
			'version' => self::FORMAT_VERSION,
			'dim'     => $this->dim,
			'count'   => count( $col['ids'] ),
			'ids'     => $col['ids'],
			'meta'    => $col['meta'],
		);

		// Atomic write with file locking for concurrency safety
		$json_path = $this->dir . '/' . $name . '.json';
		$bin_path  = $this->dir . '/' . $name . '.bin';
		$lock_path = $this->dir . '/' . $name . '.lock';
		$tmp_json  = $json_path . '.tmp';
		$tmp_bin   = $bin_path . '.tmp';

		$lock = fopen( $lock_path, 'c' );
		if ( $lock && flock( $lock, LOCK_EX ) ) {
			file_put_contents( $tmp_json, json_encode( $manifest, JSON_UNESCAPED_SLASHES ) );
			file_put_contents( $tmp_bin, $col['bin'] );

			rename( $tmp_json, $json_path );
			rename( $tmp_bin, $bin_path );

			flock( $lock, LOCK_UN );
			fclose( $lock );
		}
	}

	private function evictLru(): void {
		while ( count( $this->cache ) > $this->maxCollections ) {
			$oldest = array_key_first( $this->cache );
			if ( isset( $this->dirty[ $oldest ] ) ) {
				$this->persistCollection( $oldest );
				unset( $this->dirty[ $oldest ] );
			}
			unset( $this->cache[ $oldest ] );
		}
	}

	// ── Math ───────────────────────────────────────────────────────────

	/**
	 * Pack a float array into binary string efficiently.
	 */
	private static function packVector( array $v, int $dim ): string {
		$bin = '';
		$count = min( count( $v ), $dim );
		for ( $i = 0; $i < $count; $i++ ) {
			$bin .= pack( 'f', $v[ $i ] );
		}
		// Pad if vector is shorter than dim
		for ( $i = $count; $i < $dim; $i++ ) {
			$bin .= pack( 'f', 0.0 );
		}
		return $bin;
	}

	/**
	 * L2-normalize a vector.
	 */
	public static function normalize( array $v ): array {
		$norm = 0.0;
		foreach ( $v as $x ) {
			$norm += $x * $x;
		}
		$norm = sqrt( $norm );
		if ( $norm <= 0 ) return $v;
		return array_map( fn( $x ) => $x / $norm, $v );
	}

	/**
	 * Cosine similarity between two float arrays.
	 */
	public static function cosineSim( array $a, array $b, int $dims ): float {
		$dot    = 0.0;
		$norm_a = 0.0;
		$norm_b = 0.0;

		for ( $i = 0; $i < $dims; $i++ ) {
			$ai = $a[ $i ];
			$bi = $b[ $i ];
			$dot    += $ai * $bi;
			$norm_a += $ai * $ai;
			$norm_b += $bi * $bi;
		}

		$denom = sqrt( $norm_a ) * sqrt( $norm_b );
		return $denom > 0 ? $dot / $denom : 0.0;
	}

	/**
	 * Euclidean distance between two vectors.
	 */
	public static function euclideanDist( array $a, array $b, int $dims ): float {
		$sum = 0.0;
		for ( $i = 0; $i < $dims; $i++ ) {
			$d = $a[ $i ] - $b[ $i ];
			$sum += $d * $d;
		}
		return sqrt( $sum );
	}

	/**
	 * Dot product between two vectors.
	 */
	public static function dotProduct( array $a, array $b, int $dims ): float {
		$dot = 0.0;
		for ( $i = 0; $i < $dims; $i++ ) {
			$dot += $a[ $i ] * $b[ $i ];
		}
		return $dot;
	}

	/**
	 * Manhattan distance between two vectors.
	 */
	public static function manhattanDist( array $a, array $b, int $dims ): float {
		$sum = 0.0;
		for ( $i = 0; $i < $dims; $i++ ) {
			$sum += abs( $a[ $i ] - $b[ $i ] );
		}
		return $sum;
	}

	/**
	 * Compute a similarity score using the given distance metric.
	 * Higher = more similar.
	 */
	public static function computeScore( array $a, array $b, int $dims, Distance $distance ): float {
		return match ( $distance ) {
			Distance::Cosine     => self::cosineSim( $a, $b, $dims ),
			Distance::DotProduct => self::dotProduct( $a, $b, $dims ),
			Distance::Euclidean  => 1.0 / ( 1.0 + self::euclideanDist( $a, $b, $dims ) ),
			Distance::Manhattan  => 1.0 / ( 1.0 + self::manhattanDist( $a, $b, $dims ) ),
		};
	}
}
