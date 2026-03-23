<?php
/**
 * IVF (Inverted File Index) — partitions vectors into clusters for sub-linear search.
 *
 * Instead of comparing the query against every vector (O(N)), IVF:
 * 1. Partitions N vectors into K clusters via k-means
 * 2. At query time, finds the P closest clusters (probes)
 * 3. Only searches vectors in those P clusters
 * 4. Scans N*(P/K) vectors instead of N → speedup ≈ K/P
 *
 * Storage format:
 *   {dir}/{collection}.ivf.json — centroids + cluster assignments
 *   Vectors remain in the base VectorStore (.bin/.json)
 *
 * Usage:
 *   $store = new VectorStore('/path', 768);
 *   $ivf = new IVFIndex($store, numClusters: 100, numProbes: 10);
 *   // After adding vectors to $store:
 *   $ivf->build('my_collection');  // K-means clustering
 *   $results = $ivf->search('my_collection', $query, 5);
 *
 * @package PHPVectorStore
 */

namespace PHPVectorStore;

class IVFIndex {

	private const FORMAT_VERSION = 1;
	private const MAX_KMEANS_ITERS = 20;

	private StoreInterface $store;
	private int $numClusters;
	private int $numProbes;

	/** @var array<string, array{centroids: float[][], assignments: int[]}> */
	private array $indices = array();

	/**
	 * @param StoreInterface $store       The underlying vector store.
	 * @param int            $numClusters Number of partitions (K). Rule of thumb: sqrt(N).
	 * @param int            $numProbes   Clusters to search at query time (P). More = more accurate, slower.
	 */
	public function __construct( StoreInterface $store, int $numClusters = 100, int $numProbes = 10 ) {
		$this->store       = $store;
		$this->numClusters = $numClusters;
		$this->numProbes   = $numProbes;
	}

	/**
	 * Build the IVF index for a collection using k-means clustering.
	 *
	 * @param string $collection Collection name.
	 * @param int    $sampleDims Dimensions used for clustering (lower = faster build).
	 * @return array Stats about the built index.
	 */
	public function build( string $collection, int $sampleDims = 128 ): array {
		$start = microtime( true );
		$ids   = $this->store->ids( $collection );
		$n     = count( $ids );

		if ( 0 === $n ) {
			return array( 'error' => 'Collection is empty.' );
		}

		$k    = min( $this->numClusters, (int) ceil( sqrt( $n ) ), $n );
		$dims = min( $sampleDims, $this->store->dimensions() );

		// Load all vectors at reduced dimensions for clustering
		$vectors = array();
		foreach ( $ids as $id ) {
			$rec = $this->store->get( $collection, $id );
			if ( $rec ) {
				$vectors[] = array_slice( $rec['vector'], 0, $dims );
			}
		}

		// K-means
		$centroids   = $this->initCentroids( $vectors, $k, $dims );
		$assignments = array();

		for ( $iter = 0; $iter < self::MAX_KMEANS_ITERS; $iter++ ) {
			// Assign each vector to nearest centroid
			$new_assignments = array();
			$changed         = 0;

			for ( $i = 0; $i < $n; $i++ ) {
				$best_c    = 0;
				$best_dist = PHP_FLOAT_MAX;

				for ( $c = 0; $c < $k; $c++ ) {
					$dist = $this->sqDist( $vectors[ $i ], $centroids[ $c ], $dims );
					if ( $dist < $best_dist ) {
						$best_dist = $dist;
						$best_c    = $c;
					}
				}

				$new_assignments[ $i ] = $best_c;
				if ( ! isset( $assignments[ $i ] ) || $assignments[ $i ] !== $best_c ) {
					$changed++;
				}
			}

			$assignments = $new_assignments;

			// Convergence check
			if ( $changed === 0 ) {
				break;
			}

			// Recompute centroids
			$sums   = array_fill( 0, $k, array_fill( 0, $dims, 0.0 ) );
			$counts = array_fill( 0, $k, 0 );

			for ( $i = 0; $i < $n; $i++ ) {
				$c = $assignments[ $i ];
				$counts[ $c ]++;
				for ( $d = 0; $d < $dims; $d++ ) {
					$sums[ $c ][ $d ] += $vectors[ $i ][ $d ];
				}
			}

			for ( $c = 0; $c < $k; $c++ ) {
				if ( $counts[ $c ] > 0 ) {
					for ( $d = 0; $d < $dims; $d++ ) {
						$centroids[ $c ][ $d ] = $sums[ $c ][ $d ] / $counts[ $c ];
					}
				}
			}
		}

		// Build cluster → IDs mapping
		$cluster_ids = array_fill( 0, $k, array() );
		for ( $i = 0; $i < $n; $i++ ) {
			$cluster_ids[ $assignments[ $i ] ][] = $ids[ $i ];
		}

		// Store index
		$index = array(
			'centroids'    => $centroids,
			'cluster_ids'  => $cluster_ids,
			'k'            => $k,
			'dims'         => $dims,
			'n'            => $n,
		);

		$this->indices[ $collection ] = $index;
		$this->persistIndex( $collection, $index );

		$build_ms = round( ( microtime( true ) - $start ) * 1000, 1 );

		// Cluster size stats
		$sizes       = array_map( 'count', $cluster_ids );
		$non_empty   = array_filter( $sizes );

		return array(
			'collection'     => $collection,
			'vectors'        => $n,
			'clusters'       => $k,
			'cluster_dims'   => $dims,
			'build_ms'       => $build_ms,
			'kmeans_iters'   => min( $iter + 1, self::MAX_KMEANS_ITERS ),
			'avg_cluster'    => count( $non_empty ) > 0 ? round( array_sum( $non_empty ) / count( $non_empty ), 1 ) : 0,
			'min_cluster'    => ! empty( $non_empty ) ? min( $non_empty ) : 0,
			'max_cluster'    => ! empty( $non_empty ) ? max( $non_empty ) : 0,
			'empty_clusters' => count( $sizes ) - count( $non_empty ),
		);
	}

	/**
	 * Search using the IVF index.
	 *
	 * @param string  $collection Collection name.
	 * @param float[] $query      Query vector.
	 * @param int     $limit      Max results.
	 * @param int     $dimSlice   Dimensions for final comparison (0 = full).
	 * @return array<array{id: string, score: float, cluster: int, metadata: array}>
	 */
	public function search( string $collection, array $query, int $limit = 5, int $dimSlice = 0 ): array {
		$index = $this->loadIndex( $collection );

		if ( ! $index ) {
			// Fallback to brute-force
			return $this->store->search( $collection, $query, $limit, $dimSlice );
		}

		$q    = VectorStore::normalize( $query );
		$k    = $index['k'];
		$dims = $index['dims'];

		// Find nearest clusters
		$cluster_dists = array();
		for ( $c = 0; $c < $k; $c++ ) {
			$dist = $this->sqDist( array_slice( $q, 0, $dims ), $index['centroids'][ $c ], $dims );
			$cluster_dists[] = array( 'cluster' => $c, 'dist' => $dist );
		}
		usort( $cluster_dists, fn( $a, $b ) => $a['dist'] <=> $b['dist'] );

		// Probe top P clusters
		$probes     = min( $this->numProbes, $k );
		$candidates = array();
		$full_dims  = $dimSlice > 0 ? min( $dimSlice, $this->store->dimensions() ) : $this->store->dimensions();

		for ( $p = 0; $p < $probes; $p++ ) {
			$c   = $cluster_dists[ $p ]['cluster'];
			$ids = $index['cluster_ids'][ $c ] ?? array();

			foreach ( $ids as $id ) {
				$rec = $this->store->get( $collection, $id );
				if ( ! $rec ) continue;

				$score = VectorStore::cosineSim( $q, $rec['vector'], $full_dims );
				if ( $score > 0 ) {
					$candidates[] = array(
						'id'       => $id,
						'score'    => round( $score, 6 ),
						'cluster'  => $c,
						'metadata' => $rec['metadata'],
					);
				}
			}
		}

		usort( $candidates, fn( $a, $b ) => $b['score'] <=> $a['score'] );
		return array_slice( $candidates, 0, $limit );
	}

	/**
	 * IVF + Matryoshka combined search.
	 *
	 * Uses IVF to narrow clusters, then Matryoshka stages within candidates.
	 */
	public function matryoshkaSearch(
		string $collection,
		array  $query,
		int    $limit = 5,
		array  $stages = array( 128, 384, 768 ),
		int    $candidateMultiplier = 3
	): array {
		$index = $this->loadIndex( $collection );

		if ( ! $index ) {
			return $this->store->matryoshkaSearch( $collection, $query, $limit, $stages, $candidateMultiplier );
		}

		$q    = VectorStore::normalize( $query );
		$k    = $index['k'];
		$dims = $index['dims'];

		// Find nearest clusters
		$cluster_dists = array();
		for ( $c = 0; $c < $k; $c++ ) {
			$dist = $this->sqDist( array_slice( $q, 0, $dims ), $index['centroids'][ $c ], $dims );
			$cluster_dists[] = array( 'cluster' => $c, 'dist' => $dist );
		}
		usort( $cluster_dists, fn( $a, $b ) => $a['dist'] <=> $b['dist'] );

		// Collect candidate IDs from top P clusters
		$probes       = min( $this->numProbes, $k );
		$candidate_ids = array();
		for ( $p = 0; $p < $probes; $p++ ) {
			$c   = $cluster_dists[ $p ]['cluster'];
			$ids = $index['cluster_ids'][ $c ] ?? array();
			foreach ( $ids as $id ) {
				$candidate_ids[ $id ] = $c;
			}
		}

		if ( empty( $candidate_ids ) ) {
			return array();
		}

		// Matryoshka stages on candidates only
		sort( $stages );
		$stages = array_map( fn( $s ) => min( $s, $this->store->dimensions() ), $stages );

		$survivors    = array_keys( $candidate_ids );
		$stage_scores = array();

		foreach ( $stages as $si => $stage_dims ) {
			$is_last = $si === count( $stages ) - 1;
			$keep    = $is_last ? $limit : $limit * $candidateMultiplier;

			$scored = array();
			foreach ( $survivors as $id ) {
				$rec = $this->store->get( $collection, $id );
				if ( ! $rec ) continue;

				$score = VectorStore::cosineSim( $q, $rec['vector'], $stage_dims );
				$scored[] = array( 'id' => $id, 'score' => $score );

				if ( ! isset( $stage_scores[ $id ] ) ) {
					$stage_scores[ $id ] = array();
				}
				$stage_scores[ $id ][ $stage_dims ] = round( $score, 6 );
			}

			usort( $scored, fn( $a, $b ) => $b['score'] <=> $a['score'] );
			$scored    = array_slice( $scored, 0, $keep );
			$survivors = array_map( fn( $s ) => $s['id'], $scored );
		}

		// Build results
		$last_dim = end( $stages );
		$results  = array();
		foreach ( $survivors as $id ) {
			$results[] = array(
				'id'       => $id,
				'score'    => $stage_scores[ $id ][ $last_dim ] ?? 0.0,
				'stages'   => $stage_scores[ $id ] ?? array(),
				'cluster'  => $candidate_ids[ $id ] ?? -1,
				'metadata' => $this->store->get( $collection, $id )['metadata'] ?? array(),
			);
		}

		usort( $results, fn( $a, $b ) => $b['score'] <=> $a['score'] );
		return array_slice( $results, 0, $limit );
	}

	/**
	 * Check if an index exists for a collection.
	 */
	public function hasIndex( string $collection ): bool {
		return null !== $this->loadIndex( $collection );
	}

	/**
	 * Get index stats.
	 */
	public function indexStats( string $collection ): ?array {
		$index = $this->loadIndex( $collection );
		if ( ! $index ) return null;

		$sizes = array_map( 'count', $index['cluster_ids'] );
		return array(
			'clusters'      => $index['k'],
			'vectors'       => $index['n'],
			'cluster_dims'  => $index['dims'],
			'probes'        => $this->numProbes,
			'scan_ratio'    => round( $this->numProbes / $index['k'], 3 ),
			'avg_cluster'   => round( array_sum( $sizes ) / max( count( $sizes ), 1 ), 1 ),
		);
	}

	/**
	 * Drop the index (forces fallback to brute-force).
	 */
	public function dropIndex( string $collection ): void {
		$path = $this->indexPath( $collection );
		if ( file_exists( $path ) ) unlink( $path );
		unset( $this->indices[ $collection ] );
	}

	// ── Private ────────────────────────────────────────────────────────

	private function initCentroids( array $vectors, int $k, int $dims ): array {
		// K-means++ initialization
		$n         = count( $vectors );
		$centroids = array();

		// First centroid: random
		$centroids[] = $vectors[ mt_rand( 0, $n - 1 ) ];

		for ( $c = 1; $c < $k; $c++ ) {
			// Compute distances to nearest existing centroid
			$dists   = array();
			$sum     = 0.0;
			for ( $i = 0; $i < $n; $i++ ) {
				$min_dist = PHP_FLOAT_MAX;
				for ( $j = 0; $j < $c; $j++ ) {
					$d = $this->sqDist( $vectors[ $i ], $centroids[ $j ], $dims );
					if ( $d < $min_dist ) $min_dist = $d;
				}
				$dists[ $i ] = $min_dist;
				$sum += $min_dist;
			}

			// Weighted random selection
			if ( $sum <= 0 ) {
				$centroids[] = $vectors[ mt_rand( 0, $n - 1 ) ];
				continue;
			}

			$threshold = ( mt_rand() / mt_getrandmax() ) * $sum;
			$running   = 0.0;
			for ( $i = 0; $i < $n; $i++ ) {
				$running += $dists[ $i ];
				if ( $running >= $threshold ) {
					$centroids[] = $vectors[ $i ];
					break;
				}
			}

			if ( count( $centroids ) <= $c ) {
				$centroids[] = $vectors[ mt_rand( 0, $n - 1 ) ];
			}
		}

		return $centroids;
	}

	private function sqDist( array $a, array $b, int $dims ): float {
		$sum = 0.0;
		for ( $i = 0; $i < $dims; $i++ ) {
			$d = ( $a[ $i ] ?? 0.0 ) - ( $b[ $i ] ?? 0.0 );
			$sum += $d * $d;
		}
		return $sum;
	}

	private function loadIndex( string $collection ): ?array {
		if ( isset( $this->indices[ $collection ] ) ) {
			return $this->indices[ $collection ];
		}

		$path = $this->indexPath( $collection );
		if ( ! file_exists( $path ) ) {
			return null;
		}

		$data = json_decode( file_get_contents( $path ), true );
		if ( ! $data || ( $data['version'] ?? 0 ) !== self::FORMAT_VERSION ) {
			return null;
		}

		$this->indices[ $collection ] = $data['index'];
		return $data['index'];
	}

	private function persistIndex( string $collection, array $index ): void {
		$path = $this->indexPath( $collection );
		$data = array(
			'version' => self::FORMAT_VERSION,
			'built'   => date( 'c' ),
			'index'   => $index,
		);

		$tmp = $path . '.tmp';
		file_put_contents( $tmp, json_encode( $data, JSON_UNESCAPED_SLASHES ) );
		rename( $tmp, $path );
	}

	private function indexPath( string $collection ): string {
		return $this->store->directory() . '/' . $collection . '.ivf.json';
	}
}
