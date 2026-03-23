<?php

namespace PHPVectorStore\Tests;

use PHPUnit\Framework\TestCase;
use PHPVectorStore\VectorStore;
use PHPVectorStore\StoreInterface;
use PHPVectorStore\Distance;

class VectorStoreTest extends TestCase
{
	private string $dir;

	protected function setUp(): void {
		$this->dir = sys_get_temp_dir() . '/phpvs_test_' . uniqid();
		mkdir( $this->dir, 0755, true );
	}

	protected function tearDown(): void {
		$this->removeDir( $this->dir );
	}

	public function testImplementsStoreInterface(): void {
		$store = new VectorStore( $this->dir, 4 );
		$this->assertInstanceOf( StoreInterface::class, $store );
	}

	public function testSetAndGet(): void {
		$store = new VectorStore( $this->dir, 4 );
		$store->set( 'test', 'doc1', [1.0, 0.0, 0.0, 0.0], ['title' => 'Hello'] );

		$this->assertTrue( $store->has( 'test', 'doc1' ) );
		$this->assertEquals( 1, $store->count( 'test' ) );

		$result = $store->get( 'test', 'doc1' );
		$this->assertNotNull( $result );
		$this->assertEquals( 'doc1', $result['id'] );
		$this->assertEquals( ['title' => 'Hello'], $result['metadata'] );
	}

	public function testRemove(): void {
		$store = new VectorStore( $this->dir, 4 );
		$store->set( 'test', 'doc1', [1, 0, 0, 0] );
		$store->set( 'test', 'doc2', [0, 1, 0, 0] );

		$this->assertTrue( $store->remove( 'test', 'doc1' ) );
		$this->assertFalse( $store->has( 'test', 'doc1' ) );
		$this->assertEquals( 1, $store->count( 'test' ) );
	}

	public function testSearchCosine(): void {
		$store = new VectorStore( $this->dir, 4 );
		$store->set( 'test', 'close',  [1.0, 0.1, 0.0, 0.0] );
		$store->set( 'test', 'far',    [0.0, 0.0, 0.0, 1.0] );
		$store->set( 'test', 'medium', [0.5, 0.5, 0.0, 0.0] );

		$results = $store->search( 'test', [1.0, 0.0, 0.0, 0.0], 3 );
		$this->assertGreaterThanOrEqual( 2, count( $results ) );
		$this->assertEquals( 'close', $results[0]['id'] );
	}

	public function testSearchEuclidean(): void {
		$store = new VectorStore( $this->dir, 4 );
		$store->set( 'test', 'close', [1.0, 0.0, 0.0, 0.0] );
		$store->set( 'test', 'far',   [0.0, 0.0, 0.0, 1.0] );

		$results = $store->search( 'test', [1.0, 0.0, 0.0, 0.0], 2, 0, Distance::Euclidean );
		$this->assertEquals( 'close', $results[0]['id'] );
	}

	public function testSearchDotProduct(): void {
		$store = new VectorStore( $this->dir, 4 );
		$store->set( 'test', 'aligned', [1.0, 0.0, 0.0, 0.0] );
		$store->set( 'test', 'ortho',   [0.0, 1.0, 0.0, 0.0] );

		$results = $store->search( 'test', [1.0, 0.0, 0.0, 0.0], 2, 0, Distance::DotProduct );
		$this->assertEquals( 'aligned', $results[0]['id'] );
	}

	public function testMatryoshkaSearch(): void {
		$store = new VectorStore( $this->dir, 8 );
		for ( $i = 0; $i < 20; $i++ ) {
			$v = array_fill( 0, 8, 0.0 );
			$v[ $i % 8 ] = 1.0;
			$v[0] += 0.01 * $i;
			$store->set( 'test', "doc_$i", $v );
		}

		$results = $store->matryoshkaSearch( 'test', [1, 0, 0, 0, 0, 0, 0, 0], 3, [4, 8] );
		$this->assertCount( 3, $results );
		$this->assertArrayHasKey( 'stages', $results[0] );
	}

	public function testSearchAcross(): void {
		$store = new VectorStore( $this->dir, 4 );
		$store->set( 'col_a', 'a1', [1, 0, 0, 0] );
		$store->set( 'col_b', 'b1', [0.9, 0.1, 0, 0] );

		$results = $store->searchAcross( ['col_a', 'col_b'], [1, 0, 0, 0], 5 );
		$this->assertGreaterThanOrEqual( 2, count( $results ) );
		$this->assertArrayHasKey( 'collection', $results[0] );
	}

	public function testFlushAndReload(): void {
		$store = new VectorStore( $this->dir, 4 );
		$store->set( 'test', 'doc1', [1, 0, 0, 0], ['key' => 'value'] );
		$store->flush();

		$store2 = new VectorStore( $this->dir, 4 );
		$result = $store2->get( 'test', 'doc1' );
		$this->assertNotNull( $result );
		$this->assertEquals( ['key' => 'value'], $result['metadata'] );
	}

	public function testImportExport(): void {
		$store = new VectorStore( $this->dir, 4 );
		$records = [
			['id' => 'r1', 'vector' => [1, 0, 0, 0], 'metadata' => ['a' => 1]],
			['id' => 'r2', 'vector' => [0, 1, 0, 0]],
		];

		$this->assertEquals( 2, $store->import( 'test', $records ) );
		$this->assertCount( 2, $store->export( 'test' ) );
	}

	public function testCollections(): void {
		$store = new VectorStore( $this->dir, 4 );
		$store->set( 'alpha', 'a1', [1, 0, 0, 0] );
		$store->set( 'beta', 'b1', [0, 1, 0, 0] );
		$store->flush();

		$collections = $store->collections();
		$this->assertContains( 'alpha', $collections );
		$this->assertContains( 'beta', $collections );
	}

	public function testComputeScore(): void {
		$a = [1.0, 0.0, 0.0];
		$b = [1.0, 0.0, 0.0];
		$this->assertEqualsWithDelta( 1.0, VectorStore::computeScore( $a, $b, 3, Distance::Cosine ), 0.01 );
		$this->assertEqualsWithDelta( 1.0, VectorStore::computeScore( $a, $b, 3, Distance::Euclidean ), 0.01 );
	}

	public function testEmptySearch(): void {
		$store = new VectorStore( $this->dir, 4 );
		$this->assertEmpty( $store->search( 'empty', [1, 0, 0, 0] ) );
	}

	private function removeDir( string $dir ): void {
		if ( ! is_dir( $dir ) ) return;
		foreach ( glob( $dir . '/*' ) ?: [] as $file ) {
			is_dir( $file ) ? $this->removeDir( $file ) : unlink( $file );
		}
		rmdir( $dir );
	}
}
