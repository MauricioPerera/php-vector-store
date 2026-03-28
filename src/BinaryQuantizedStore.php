<?php
/**
 * Binary Quantization Store — 1-bit vectors for 32x compression and ultra-fast Hamming search.
 *
 * Converts 768 × float32 (3072 bytes) → 768 bits = 96 bytes per vector.
 * Each float is reduced to its sign bit: value >= 0 → 1, value < 0 → 0.
 *
 * Similarity is computed via Hamming distance (XOR + popcount) and converted
 * to an approximate cosine similarity: 1.0 - 2.0 * hamming / dims.
 *
 * Accuracy loss: ~5-10% on cosine similarity. Best combined with Matryoshka
 * multi-stage search for accuracy recovery.
 *
 * Storage format:
 *   {dir}/{collection}.b1.bin  — packed bits, ceil(dim/8) bytes per vector
 *   {dir}/{collection}.b1.json — manifest with IDs, metadata
 *
 * @package PHPVectorStore
 */

namespace PHPVectorStore;

class BinaryQuantizedStore implements StoreInterface {

	private const FORMAT_VERSION = 1;

	private int    $dim;
	private string $dir;

	/** @var array<string, array{ids: string[], id_map: array<string,int>, bin: string, meta: array}> */
	private array $cache = array();
	private array $dirty = array();

	/** @var int[] Popcount lookup table for bytes 0-255. */
	private static array $POPCOUNT = array();

	public function __construct( string $directory, int $dimensions = 768 ) {
		$this->dir = rtrim( $directory, '/\\' );
		$this->dim = $dimensions;

		if ( ! is_dir( $this->dir ) ) {
			mkdir( $this->dir, 0755, true );
		}

		if ( empty( self::$POPCOUNT ) ) {
			for ( $i = 0; $i < 256; $i++ ) {
				self::$POPCOUNT[ $i ] = 0;
				$n = $i;
				while ( $n ) {
					self::$POPCOUNT[ $i ]++;
					$n &= $n - 1;
				}
			}
		}
	}

	public function dimensions(): int {
		return $this->dim;
	}

	public function directory(): string {
		return $this->dir;
	}

	/**
	 * Bytes per binary-quantized vector: ceil(dim / 8).
	 */
	public function bytesPerVector(): int {
		return (int) ceil( $this->dim / 8 );
	}

	// ── Write ──────────────────────────────────────────────────────────

	/**
	 * Store a vector (quantized to 1-bit sign).
	 *
	 * @param string  $collection Collection name.
	 * @param string  $id         Unique ID.
	 * @param float[] $vector     Float vector (will be normalized then binary-quantized).
	 * @param array   $metadata   Optional metadata.
	 */
	public function set( string $collection, string $id, array $vector, array $metadata = array() ): void {
		if ( count( $vector ) < $this->dim ) {
			throw Exception\DimensionMismatchException::forVectors( $this->dim, count( $vector ) );
		}

		$col = $this->loadCollection( $collection );

		$normalized = VectorStore::normalize( $vector );
		$binary     = self::quantize( $normalized, $this->dim );

		$bpv = $this->bytesPerVector();
		$pos = $col['id_map'][ $id ] ?? null;

		if ( null !== $pos ) {
			$col['bin'] = substr_replace( $col['bin'], $binary, $pos * $bpv, $bpv );
			$col['meta'][ $id ] = $metadata;
		} else {
			$col['ids'][]          = $id;
			$col['id_map'][ $id ]  = count( $col['ids'] ) - 1;
			$col['bin']           .= $binary;
			$col['meta'][ $id ]    = $metadata;
		}

		$this->cache[ $collection ] = $col;
		$this->dirty[ $collection ] = true;
	}

	/**
	 * Remove a vector.
	 */
	public function remove( string $collection, string $id ): bool {
		$col = $this->loadCollection( $collection );
		$pos = $col['id_map'][ $id ] ?? null;
		if ( null === $pos ) return false;

		$bpv      = $this->bytesPerVector();
		$last_idx = count( $col['ids'] ) - 1;

		if ( $pos !== $last_idx ) {
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

	// ── Read ───────────────────────────────────────────────────────────

	public function get( string $collection, string $id ): ?array {
		$col = $this->loadCollection( $collection );
		$pos = $col['id_map'][ $id ] ?? null;
		if ( null === $pos ) return null;

		$bpv    = $this->bytesPerVector();
		$offset = $pos * $bpv;
		$floats = self::dequantize( $col['bin'], $offset, $this->dim );

		return array( 'id' => $id, 'vector' => $floats, 'metadata' => $col['meta'][ $id ] ?? array() );
	}

	public function has( string $collection, string $id ): bool {
		$col = $this->loadCollection( $collection );
		return isset( $col['id_map'][ $id ] );
	}

	public function count( string $collection ): int {
		$col = $this->loadCollection( $collection );
		return count( $col['ids'] );
	}

	public function ids( string $collection ): array {
		$col = $this->loadCollection( $collection );
		return $col['ids'];
	}

	public function collections(): array {
		$files = glob( $this->dir . '/*.b1.json' );
		return array_map( fn( $f ) => basename( $f, '.b1.json' ), $files ?: array() );
	}

	// ── Search ─────────────────────────────────────────────────────────

	/**
	 * Search using binary Hamming distance (XOR + popcount).
	 */
	public function search( string $collection, array $query, int $limit = 5, int $dimSlice = 0, ?Distance $distance = null ): array {
		$col = $this->loadCollection( $collection );
		if ( empty( $col['ids'] ) ) return array();

		$distance = $distance ?? Distance::Cosine;
		$dims     = $dimSlice > 0 ? min( $dimSlice, $this->dim ) : $this->dim;
		$q_norm   = VectorStore::normalize( $query );
		$count    = count( $col['ids'] );
		$bpv      = $this->bytesPerVector();
		$results  = array();

		if ( $distance === Distance::Cosine ) {
			$q_binary    = self::quantize( $q_norm, $this->dim );
			$bytes_to_cmp = (int) ceil( $dims / 8 );

			for ( $i = 0; $i < $count; $i++ ) {
				$score = self::binaryCosineSim( $q_binary, 0, $col['bin'], $i * $bpv, $dims, $bytes_to_cmp );
				if ( $score > 0 ) {
					$results[] = array(
						'id'       => $col['ids'][ $i ],
						'score'    => round( $score, 6 ),
						'metadata' => $col['meta'][ $col['ids'][ $i ] ] ?? array(),
					);
				}
			}
		} else {
			for ( $i = 0; $i < $count; $i++ ) {
				$v     = self::dequantize( $col['bin'], $i * $bpv, $this->dim );
				$score = VectorStore::computeScore( $q_norm, $v, $dims, $distance );
				if ( $score > 0 ) {
					$results[] = array(
						'id'       => $col['ids'][ $i ],
						'score'    => round( $score, 6 ),
						'metadata' => $col['meta'][ $col['ids'][ $i ] ] ?? array(),
					);
				}
			}
		}

		usort( $results, fn( $a, $b ) => $b['score'] <=> $a['score'] );
		return array_slice( $results, 0, $limit );
	}

	/**
	 * Matryoshka search on binary-quantized vectors.
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
		if ( empty( $col['ids'] ) ) return array();

		$distance = $distance ?? Distance::Cosine;
		$q_norm   = VectorStore::normalize( $query );
		$q_binary = ( $distance === Distance::Cosine ) ? self::quantize( $q_norm, $this->dim ) : null;

		$bpv   = $this->bytesPerVector();
		$count = count( $col['ids'] );

		sort( $stages );
		$stages = array_map( fn( $s ) => min( $s, $this->dim ), $stages );

		$candidate_indices = range( 0, $count - 1 );
		$stage_scores      = array();

		foreach ( $stages as $si => $dims ) {
			$is_last       = $si === count( $stages ) - 1;
			$keep          = $is_last ? $limit : $limit * $candidateMultiplier;
			$bytes_to_cmp  = (int) ceil( $dims / 8 );

			$scored = array();

			foreach ( $candidate_indices as $i ) {
				if ( $q_binary !== null ) {
					$score = self::binaryCosineSim( $q_binary, 0, $col['bin'], $i * $bpv, $dims, $bytes_to_cmp );
				} else {
					$v     = self::dequantize( $col['bin'], $i * $bpv, $this->dim );
					$score = VectorStore::computeScore( $q_norm, $v, $dims, $distance );
				}

				$scored[] = array( 'index' => $i, 'score' => $score );

				$id = $col['ids'][ $i ];
				if ( ! isset( $stage_scores[ $id ] ) ) $stage_scores[ $id ] = array();
				$stage_scores[ $id ][ $dims ] = round( $score, 6 );
			}

			usort( $scored, fn( $a, $b ) => $b['score'] <=> $a['score'] );
			$scored            = array_slice( $scored, 0, $keep );
			$candidate_indices = array_map( fn( $s ) => $s['index'], $scored );
		}

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

	// ── Persistence ────────────────────────────────────────────────────

	public function flush(): void {
		foreach ( $this->dirty as $col => $_ ) {
			$this->persistCollection( $col );
		}
		$this->dirty = array();
	}

	public function drop( string $collection ): void {
		@unlink( $this->dir . '/' . $collection . '.b1.json' );
		@unlink( $this->dir . '/' . $collection . '.b1.bin' );
		unset( $this->cache[ $collection ], $this->dirty[ $collection ] );
	}

	public function stats(): array {
		$total_vectors = 0;
		$total_bytes   = 0;
		$detail        = array();

		foreach ( $this->collections() as $name ) {
			$bin  = $this->dir . '/' . $name . '.b1.bin';
			$size = file_exists( $bin ) ? filesize( $bin ) : 0;
			$vecs = $size > 0 ? (int) ( $size / $this->bytesPerVector() ) : 0;

			$total_vectors += $vecs;
			$total_bytes   += $size;
			$detail[]       = array( 'collection' => $name, 'vectors' => $vecs, 'bytes' => $size );
		}

		return array(
			'dimensions'    => $this->dim,
			'quantization'  => 'binary1bit',
			'total_vectors' => $total_vectors,
			'total_bytes'   => $total_bytes,
			'memory_mb'     => round( $total_bytes / ( 1024 * 1024 ), 3 ),
			'bytes_per_vec' => $this->bytesPerVector(),
			'compression'   => round( ( $this->dim * 4 ) / $this->bytesPerVector(), 1 ) . 'x',
			'collections'   => $detail,
		);
	}

	// ── Multi-collection Search ───────────────────────────────────────

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

	// ── Import / Export ────────────────────────────────────────────────

	public function import( string $collection, array $records ): int {
		$count = 0;
		foreach ( $records as $r ) {
			if ( empty( $r['id'] ) || empty( $r['vector'] ) ) continue;
			$this->set( $collection, $r['id'], $r['vector'], $r['metadata'] ?? array() );
			$count++;
		}
		return $count;
	}

	public function export( string $collection ): array {
		$col    = $this->loadCollection( $collection );
		$bpv    = $this->bytesPerVector();
		$result = array();
		foreach ( $col['ids'] as $i => $id ) {
			$floats   = self::dequantize( $col['bin'], $i * $bpv, $this->dim );
			$result[] = array( 'id' => $id, 'vector' => $floats, 'metadata' => $col['meta'][ $id ] ?? array() );
		}
		return $result;
	}

	// ── Binary Quantization Math ──────────────────────────────────────

	/**
	 * Quantize float vector to binary (1-bit per dimension).
	 * Each float is reduced to its sign: value >= 0 → bit 1, value < 0 → bit 0.
	 * Bits are packed MSB-first: dimension 0 is the highest bit of byte 0.
	 */
	public static function quantize( array $vector, int $dim ): string {
		$bytes = (int) ceil( $dim / 8 );
		$bin   = str_repeat( "\0", $bytes );
		$d     = min( count( $vector ), $dim );

		for ( $i = 0; $i < $d; $i++ ) {
			if ( $vector[ $i ] >= 0 ) {
				$byte_idx = (int) ( $i / 8 );
				$bit_pos  = 7 - ( $i % 8 );  // MSB-first
				$bin[ $byte_idx ] = chr( ord( $bin[ $byte_idx ] ) | ( 1 << $bit_pos ) );
			}
		}

		return $bin;
	}

	/**
	 * Dequantize binary back to float vector.
	 * bit 1 → +1.0, bit 0 → -1.0 (sign recovery for normalized vectors).
	 */
	public static function dequantize( string $bin, int $offset, int $dim ): array {
		$floats = array();

		for ( $i = 0; $i < $dim; $i++ ) {
			$byte_idx = $offset + (int) ( $i / 8 );
			$bit_pos  = 7 - ( $i % 8 );
			$bit      = ( ord( $bin[ $byte_idx ] ) >> $bit_pos ) & 1;
			$floats[] = $bit ? 1.0 : -1.0;
		}

		return $floats;
	}

	/**
	 * Approximate cosine similarity using Hamming distance on binary vectors.
	 *
	 * For normalized vectors quantized to sign bits:
	 *   cosine_sim ≈ 1.0 - 2.0 * hamming_distance / dimensions
	 *
	 * This is exact when vectors are unit-normalized and only sign information
	 * is preserved: matching signs contribute positively, mismatches negatively.
	 */
	private static function binaryCosineSim( string $bin_a, int $offset_a, string $bin_b, int $offset_b, int $dims, int $bytes_to_cmp ): float {
		$hamming = 0;

		for ( $i = 0; $i < $bytes_to_cmp; $i++ ) {
			$xor = ord( $bin_a[ $offset_a + $i ] ) ^ ord( $bin_b[ $offset_b + $i ] );
			$hamming += self::$POPCOUNT[ $xor ];
		}

		// If dims is not byte-aligned, mask out padding bits in the last byte
		$remainder = $dims % 8;
		if ( $remainder > 0 ) {
			$last      = $bytes_to_cmp - 1;
			$xor       = ord( $bin_a[ $offset_a + $last ] ) ^ ord( $bin_b[ $offset_b + $last ] );
			$mask      = ( 0xFF << ( 8 - $remainder ) ) & 0xFF;
			$corrected = self::$POPCOUNT[ $xor & $mask ];
			$hamming   = $hamming - self::$POPCOUNT[ $xor ] + $corrected;
		}

		return 1.0 - ( 2.0 * $hamming / $dims );
	}

	// ── Private ────────────────────────────────────────────────────────

	private function loadCollection( string $name ): array {
		if ( isset( $this->cache[ $name ] ) ) return $this->cache[ $name ];

		$json_path = $this->dir . '/' . $name . '.b1.json';
		$bin_path  = $this->dir . '/' . $name . '.b1.bin';

		if ( ! file_exists( $json_path ) || ! file_exists( $bin_path ) ) {
			$col = array( 'ids' => array(), 'id_map' => array(), 'bin' => '', 'meta' => array() );
			$this->cache[ $name ] = $col;
			return $col;
		}

		$manifest = json_decode( file_get_contents( $json_path ), true );
		if ( ! $manifest || ( $manifest['version'] ?? 0 ) !== self::FORMAT_VERSION ) {
			$col = array( 'ids' => array(), 'id_map' => array(), 'bin' => '', 'meta' => array() );
			$this->cache[ $name ] = $col;
			return $col;
		}

		$id_map = array();
		foreach ( $manifest['ids'] as $i => $id ) $id_map[ $id ] = $i;

		$col = array(
			'ids'    => $manifest['ids'],
			'id_map' => $id_map,
			'bin'    => file_get_contents( $bin_path ),
			'meta'   => $manifest['meta'] ?? array(),
		);

		$this->cache[ $name ] = $col;
		return $col;
	}

	private function persistCollection( string $name ): void {
		if ( ! isset( $this->cache[ $name ] ) ) return;
		$col = $this->cache[ $name ];

		if ( empty( $col['ids'] ) ) {
			@unlink( $this->dir . '/' . $name . '.b1.json' );
			@unlink( $this->dir . '/' . $name . '.b1.bin' );
			return;
		}

		$manifest = array(
			'version' => self::FORMAT_VERSION,
			'dim'     => $this->dim,
			'quant'   => 'binary1bit',
			'count'   => count( $col['ids'] ),
			'ids'     => $col['ids'],
			'meta'    => $col['meta'],
		);

		$json_path = $this->dir . '/' . $name . '.b1.json';
		$bin_path  = $this->dir . '/' . $name . '.b1.bin';
		$lock_path = $this->dir . '/' . $name . '.b1.lock';
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
}
