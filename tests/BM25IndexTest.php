<?php

namespace PHPVectorStore\Tests;

use PHPUnit\Framework\TestCase;
use PHPVectorStore\BM25\Index;
use PHPVectorStore\BM25\Config;
use PHPVectorStore\BM25\SimpleTokenizer;

class BM25IndexTest extends TestCase
{
	private string $dir;

	protected function setUp(): void {
		$this->dir = sys_get_temp_dir() . '/phpvs_bm25_' . uniqid();
		mkdir( $this->dir, 0755, true );
	}

	protected function tearDown(): void {
		$this->removeDir( $this->dir );
	}

	public function testAddAndSearch(): void {
		$index = new Index();
		$index->addDocument( 'test', 'doc1', 'The quick brown fox jumps over the lazy dog' );
		$index->addDocument( 'test', 'doc2', 'A fast red car drives on the highway' );
		$index->addDocument( 'test', 'doc3', 'The brown fox is quick and agile' );

		$results = $index->search( 'test', 'quick brown fox', 2 );
		$this->assertCount( 2, $results );
		$ids = array_column( $results, 'id' );
		$this->assertContains( 'doc1', $ids );
		$this->assertContains( 'doc3', $ids );
	}

	public function testScoreAll(): void {
		$index = new Index();
		$index->addDocument( 'test', 'doc1', 'machine learning algorithms' );
		$index->addDocument( 'test', 'doc2', 'deep learning neural networks' );

		$scores = $index->scoreAll( 'test', 'learning' );
		$this->assertCount( 2, $scores );
		$this->assertGreaterThan( 0, $scores['doc1'] );
	}

	public function testRemoveDocument(): void {
		$index = new Index();
		$index->addDocument( 'test', 'doc1', 'hello world' );
		$index->addDocument( 'test', 'doc2', 'goodbye world' );

		$index->removeDocument( 'test', 'doc1' );
		$this->assertEquals( 1, $index->count( 'test' ) );
		$this->assertEmpty( $index->search( 'test', 'hello', 5 ) );
	}

	public function testCollectionIsolation(): void {
		$index = new Index();
		$index->addDocument( 'alpha', 'doc1', 'hello world' );
		$index->addDocument( 'beta', 'doc2', 'goodbye world' );

		$this->assertEmpty( $index->search( 'alpha', 'goodbye', 5 ) );
	}

	public function testPersistenceRoundTrip(): void {
		$index = new Index();
		$index->addDocument( 'test', 'doc1', 'vector database search' );
		$index->addDocument( 'test', 'doc2', 'full text retrieval' );
		$index->save( $this->dir, 'test' );

		$index2 = new Index();
		$index2->load( $this->dir, 'test' );
		$this->assertEquals( 2, $index2->count( 'test' ) );
		$this->assertNotEmpty( $index2->scoreAll( 'test', 'vector search' ) );
	}

	public function testConfigValidation(): void {
		$this->expectException( \InvalidArgumentException::class );
		new Config( k1: -1.0 );
	}

	public function testTokenizerStopWords(): void {
		$t = new SimpleTokenizer();
		$tokens = $t->tokenize( 'the quick brown fox and the lazy dog' );
		$this->assertNotContains( 'the', $tokens );
		$this->assertContains( 'quick', $tokens );
	}

	public function testTokenizerUnicode(): void {
		$t = new SimpleTokenizer( stopWords: [] );
		$tokens = $t->tokenize( 'café résumé' );
		$this->assertContains( 'café', $tokens );
	}

	public function testEmptyIndex(): void {
		$index = new Index();
		$this->assertEmpty( $index->search( 'test', 'hello' ) );
	}

	private function removeDir( string $dir ): void {
		if ( ! is_dir( $dir ) ) return;
		foreach ( glob( $dir . '/*' ) ?: [] as $file ) {
			is_dir( $file ) ? $this->removeDir( $file ) : unlink( $file );
		}
		rmdir( $dir );
	}
}
