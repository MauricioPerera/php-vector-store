<?php

namespace PHPVectorStore;

use PHPVectorStore\BM25\Index as BM25Index;

/**
 * Hybrid Search — combines vector similarity with BM25 full-text scoring.
 *
 * Two fusion strategies:
 * - RRF (Reciprocal Rank Fusion): rank-based, robust to score scale differences.
 * - Weighted: linear combination of min-max normalized scores.
 */
class HybridSearch
{
	public function __construct(
		private readonly StoreInterface $store,
		private readonly BM25Index $bm25,
		private readonly HybridMode $mode = HybridMode::RRF,
	) {}

	/**
	 * @param float[] $vector  Query vector.
	 * @param string  $text    Query text for BM25.
	 * @param array   $options fetchK, vectorWeight, textWeight, rrfK, dimSlice.
	 * @return array<array{id: string, score: float, metadata: array}>
	 */
	public function search( string $collection, array $vector, string $text, int $limit = 5, array $options = [] ): array {
		$fetchK       = $options['fetchK'] ?? max( $limit * 3, 50 );
		$vectorWeight = $options['vectorWeight'] ?? 0.5;
		$textWeight   = $options['textWeight'] ?? 0.5;
		$rrfK         = $options['rrfK'] ?? 60;
		$dimSlice     = $options['dimSlice'] ?? 0;

		$vectorResults = $this->store->search( $collection, $vector, $fetchK, $dimSlice );
		$textScores    = $this->bm25->scoreAll( $collection, $text );

		if ( empty( $vectorResults ) && empty( $textScores ) ) {
			return [];
		}

		return match ( $this->mode ) {
			HybridMode::RRF      => $this->fuseRRF( $vectorResults, $textScores, $limit, $rrfK ),
			HybridMode::Weighted => $this->fuseWeighted( $vectorResults, $textScores, $limit, $vectorWeight, $textWeight ),
		};
	}

	/** @return array<array{id: string, score: float, collection: string, metadata: array}> */
	public function searchAcross( array $collections, array $vector, string $text, int $limit = 5, array $options = [] ): array {
		$merged = [];
		foreach ( $collections as $col ) {
			foreach ( $this->search( $col, $vector, $text, $limit, $options ) as $r ) {
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

	private function fuseRRF( array $vectorResults, array $textScores, int $limit, int $rrfK ): array {
		$fused    = [];
		$metadata = [];

		foreach ( $vectorResults as $rank => $r ) {
			$id = $r['id'];
			$fused[ $id ]    = ( $fused[ $id ] ?? 0.0 ) + 1.0 / ( $rrfK + $rank + 1 );
			$metadata[ $id ] = $r['metadata'] ?? [];
		}

		$rank = 1;
		foreach ( $textScores as $id => $score ) {
			$fused[ $id ] = ( $fused[ $id ] ?? 0.0 ) + 1.0 / ( $rrfK + $rank );
			$rank++;
		}

		arsort( $fused );

		$results = [];
		foreach ( array_slice( $fused, 0, $limit, true ) as $id => $score ) {
			$results[] = [ 'id' => $id, 'score' => round( $score, 6 ), 'metadata' => $metadata[ $id ] ?? [] ];
		}
		return $results;
	}

	private function fuseWeighted( array $vectorResults, array $textScores, int $limit, float $vectorWeight, float $textWeight ): array {
		$vecScores = [];
		$metadata  = [];
		foreach ( $vectorResults as $r ) {
			$vecScores[ $r['id'] ] = $r['score'];
			$metadata[ $r['id'] ]  = $r['metadata'] ?? [];
		}

		$vecNorm  = self::minMaxNormalize( $vecScores );
		$textNorm = self::minMaxNormalize( $textScores );

		$allIds = array_unique( array_merge( array_keys( $vecScores ), array_keys( $textScores ) ) );
		$fused  = [];

		foreach ( $allIds as $id ) {
			$fused[ $id ] = $vectorWeight * ( $vecNorm[ $id ] ?? 0.0 )
			              + $textWeight * ( $textNorm[ $id ] ?? 0.0 );
		}

		arsort( $fused );

		$results = [];
		foreach ( array_slice( $fused, 0, $limit, true ) as $id => $score ) {
			$results[] = [ 'id' => $id, 'score' => round( $score, 6 ), 'metadata' => $metadata[ $id ] ?? [] ];
		}
		return $results;
	}

	/** @return array<string, float> */
	private static function minMaxNormalize( array $scores ): array {
		if ( empty( $scores ) ) {
			return [];
		}
		$min   = min( $scores );
		$max   = max( $scores );
		$range = $max - $min;

		if ( $range <= 0 ) {
			return array_map( fn() => 1.0, $scores );
		}

		$normalized = [];
		foreach ( $scores as $id => $score ) {
			$normalized[ $id ] = ( $score - $min ) / $range;
		}
		return $normalized;
	}
}
