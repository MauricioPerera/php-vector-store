<?php

namespace PHPVectorStore\Tests;

use PHPUnit\Framework\TestCase;
use PHPVectorStore\VectorStore;
use PHPVectorStore\HybridSearch;
use PHPVectorStore\HybridMode;
use PHPVectorStore\BM25\Index as BM25Index;

class HybridSearchTest extends TestCase
{
	private string $dir;

	protected function setUp(): void {
		$this->dir = sys_get_temp_dir() . '/phpvs_hybrid_' . uniqid();
		mkdir( $this->dir, 0755, true );
	}

	protected function tearDown(): void {
		$this->removeDir( $this->dir );
	}

	private function createTestData(): array {
		$store = new VectorStore( $this->dir, 4 );
		$bm25  = new BM25Index();

		$store->set( 'test', 'doc1', [1.0, 0.1, 0.0, 0.0], ['title' => 'ML Guide'] );
		$bm25->addDocument( 'test', 'doc1', 'machine learning algorithms neural networks deep learning' );

		$store->set( 'test', 'doc2', [0.0, 1.0, 0.1, 0.0], ['title' => 'DB Systems'] );
		$bm25->addDocument( 'test', 'doc2', 'database systems sql query optimization relational' );

		$store->set( 'test', 'doc3', [0.9, 0.2, 0.0, 0.0], ['title' => 'ML + DB'] );
		$bm25->addDocument( 'test', 'doc3', 'machine learning database vector search embeddings' );

		$store->set( 'test', 'doc4', [0.0, 0.0, 0.0, 1.0], ['title' => 'Remote ML'] );
		$bm25->addDocument( 'test', 'doc4', 'machine learning transformer models attention mechanism' );

		return [$store, $bm25];
	}

	public function testRRFSearch(): void {
		[$store, $bm25] = $this->createTestData();
		$hybrid = new HybridSearch( $store, $bm25, HybridMode::RRF );

		$results = $hybrid->search( 'test', [1.0, 0.0, 0.0, 0.0], 'machine learning', 4 );
		$this->assertNotEmpty( $results );
		$this->assertLessThanOrEqual( 4, count( $results ) );

		for ( $i = 1; $i < count( $results ); $i++ ) {
			$this->assertGreaterThanOrEqual( $results[ $i ]['score'], $results[ $i - 1 ]['score'] );
		}
	}

	public function testWeightedSearch(): void {
		[$store, $bm25] = $this->createTestData();
		$hybrid = new HybridSearch( $store, $bm25, HybridMode::Weighted );

		$results = $hybrid->search( 'test', [1.0, 0.0, 0.0, 0.0], 'machine learning', 4, [
			'vectorWeight' => 0.7,
			'textWeight'   => 0.3,
		] );
		$this->assertNotEmpty( $results );
	}

	public function testHybridFindsDoc3(): void {
		[$store, $bm25] = $this->createTestData();
		$hybrid = new HybridSearch( $store, $bm25, HybridMode::RRF );

		$results = $hybrid->search( 'test', [1.0, 0.0, 0.0, 0.0], 'machine learning database', 4 );
		$ids = array_column( $results, 'id' );
		$this->assertContains( 'doc3', $ids );
	}

	public function testEmptyBM25StillReturnsVectorResults(): void {
		[$store, $bm25] = $this->createTestData();
		$hybrid = new HybridSearch( $store, $bm25, HybridMode::RRF );

		$results = $hybrid->search( 'test', [1, 0, 0, 0], 'xyznonexistent', 5 );
		$this->assertNotEmpty( $results );
	}

	public function testSearchAcross(): void {
		$store = new VectorStore( $this->dir, 4 );
		$bm25  = new BM25Index();

		$store->set( 'col_a', 'a1', [1, 0, 0, 0] );
		$bm25->addDocument( 'col_a', 'a1', 'machine learning' );
		$store->set( 'col_b', 'b1', [0.9, 0.1, 0, 0] );
		$bm25->addDocument( 'col_b', 'b1', 'deep learning' );

		$hybrid = new HybridSearch( $store, $bm25, HybridMode::RRF );
		$results = $hybrid->searchAcross( ['col_a', 'col_b'], [1, 0, 0, 0], 'learning', 5 );
		$this->assertGreaterThanOrEqual( 2, count( $results ) );
		$this->assertArrayHasKey( 'collection', $results[0] );
	}

	public function testMetadataPreserved(): void {
		[$store, $bm25] = $this->createTestData();
		$hybrid = new HybridSearch( $store, $bm25, HybridMode::RRF );

		$results = $hybrid->search( 'test', [1, 0, 0, 0], 'machine learning', 4 );
		foreach ( $results as $r ) {
			$this->assertArrayHasKey( 'metadata', $r );
		}
	}

	private function removeDir( string $dir ): void {
		if ( ! is_dir( $dir ) ) return;
		foreach ( glob( $dir . '/*' ) ?: [] as $file ) {
			is_dir( $file ) ? $this->removeDir( $file ) : unlink( $file );
		}
		rmdir( $dir );
	}
}
