<?php

namespace PHPVectorStore\BM25;

/**
 * BM25 (Okapi BM25) inverted-index for full-text search.
 *
 * Collection-aware: all data structures are scoped per collection.
 */
final class Index
{
	/** @var array<string, array<string, array<string, int>>> collection → term → [docId → tf] */
	private array $invertedIndex = [];

	/** @var array<string, array<string, int>> collection → [docId → tokenCount] */
	private array $docLengths = [];

	/** @var array<string, int> collection → totalTokens */
	private array $totalTokens = [];

	/** @var array<string, int> collection → docCount */
	private array $docCount = [];

	public function __construct(
		private readonly Config $config = new Config(),
		private readonly TokenizerInterface $tokenizer = new SimpleTokenizer(),
	) {}

	public function addDocument( string $collection, string $id, string $text ): void {
		if ( $text === '' ) {
			return;
		}

		$tokens = $this->tokenizer->tokenize( $text );
		if ( empty( $tokens ) ) {
			return;
		}

		$tokenCount = count( $tokens );

		$this->docLengths[ $collection ][ $id ] = $tokenCount;
		$this->totalTokens[ $collection ]       = ( $this->totalTokens[ $collection ] ?? 0 ) + $tokenCount;
		$this->docCount[ $collection ]           = ( $this->docCount[ $collection ] ?? 0 ) + 1;

		$termFreqs = array_count_values( $tokens );
		foreach ( $termFreqs as $term => $tf ) {
			$this->invertedIndex[ $collection ][ $term ][ $id ] = $tf;
		}
	}

	public function removeDocument( string $collection, string $id ): void {
		if ( ! isset( $this->docLengths[ $collection ][ $id ] ) ) {
			return;
		}

		$tokenCount = $this->docLengths[ $collection ][ $id ];
		$this->totalTokens[ $collection ] -= $tokenCount;
		$this->docCount[ $collection ]--;

		unset( $this->docLengths[ $collection ][ $id ] );

		foreach ( $this->invertedIndex[ $collection ] as $term => &$postings ) {
			unset( $postings[ $id ] );
			if ( empty( $postings ) ) {
				unset( $this->invertedIndex[ $collection ][ $term ] );
			}
		}
		unset( $postings );
	}

	/** @return array<array{id: string, score: float, rank: int}> */
	public function search( string $collection, string $query, int $limit = 10 ): array {
		$scores = $this->scoreAll( $collection, $query );
		if ( empty( $scores ) ) {
			return [];
		}

		$results = [];
		$rank    = 1;
		foreach ( array_slice( $scores, 0, $limit, true ) as $id => $score ) {
			$results[] = [ 'id' => $id, 'score' => $score, 'rank' => $rank++ ];
		}
		return $results;
	}

	/** @return array<string, float> docId → score (sorted descending) */
	public function scoreAll( string $collection, string $query ): array {
		$numDocs = $this->docCount[ $collection ] ?? 0;
		if ( $numDocs === 0 ) {
			return [];
		}

		$queryTokens = $this->tokenizer->tokenize( $query );
		if ( empty( $queryTokens ) ) {
			return [];
		}

		$avgDl  = $this->totalTokens[ $collection ] / $numDocs;
		$scores = [];

		foreach ( array_unique( $queryTokens ) as $term ) {
			$postings = $this->invertedIndex[ $collection ][ $term ] ?? null;
			if ( $postings === null ) {
				continue;
			}

			$df  = count( $postings );
			$idf = log( ( $numDocs - $df + 0.5 ) / ( $df + 0.5 ) + 1.0 );
			$k1  = $this->config->k1;
			$b   = $this->config->b;

			foreach ( $postings as $docId => $tf ) {
				$dl     = $this->docLengths[ $collection ][ $docId ];
				$tfNorm = ( $tf * ( $k1 + 1.0 ) )
					/ ( $tf + $k1 * ( 1.0 - $b + $b * $dl / $avgDl ) );
				$scores[ $docId ] = ( $scores[ $docId ] ?? 0.0 ) + $idf * $tfNorm;
			}
		}

		arsort( $scores );
		return $scores;
	}

	public function count( string $collection ): int {
		return $this->docCount[ $collection ] ?? 0;
	}

	public function vocabularySize( string $collection ): int {
		return count( $this->invertedIndex[ $collection ] ?? [] );
	}

	public function exportState( string $collection ): array {
		return [
			'totalTokens'   => $this->totalTokens[ $collection ] ?? 0,
			'docCount'      => $this->docCount[ $collection ] ?? 0,
			'docLengths'    => $this->docLengths[ $collection ] ?? [],
			'invertedIndex' => $this->invertedIndex[ $collection ] ?? [],
		];
	}

	public function importState( string $collection, array $state ): void {
		$this->totalTokens[ $collection ]   = $state['totalTokens'] ?? 0;
		$this->docCount[ $collection ]      = $state['docCount'] ?? 0;
		$this->docLengths[ $collection ]    = $state['docLengths'] ?? [];
		$this->invertedIndex[ $collection ] = $state['invertedIndex'] ?? [];
	}

	public function save( string $directory, string $collection ): void {
		$state = $this->exportState( $collection );
		if ( empty( $state['docLengths'] ) ) {
			return;
		}
		$path = rtrim( $directory, '/\\' ) . '/' . $collection . '.bm25.bin';
		$tmp  = $path . '.tmp';
		file_put_contents( $tmp, serialize( $state ) );
		rename( $tmp, $path );
	}

	public function load( string $directory, string $collection ): void {
		$path = rtrim( $directory, '/\\' ) . '/' . $collection . '.bm25.bin';
		if ( ! file_exists( $path ) ) {
			return;
		}
		$state = unserialize( file_get_contents( $path ) );
		if ( is_array( $state ) ) {
			$this->importState( $collection, $state );
		}
	}
}
