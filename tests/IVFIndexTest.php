<?php

namespace PHPVectorStore\Tests;

use PHPUnit\Framework\TestCase;
use PHPVectorStore\VectorStore;
use PHPVectorStore\QuantizedStore;
use PHPVectorStore\IVFIndex;
use PHPVectorStore\StoreInterface;

class IVFIndexTest extends TestCase
{
	private string $dir;

	protected function setUp(): void {
		$this->dir = sys_get_temp_dir() . '/phpvs_ivf_' . uniqid();
		mkdir( $this->dir, 0755, true );
	}

	protected function tearDown(): void {
		$this->removeDir( $this->dir );
	}

	public function testBuildAndSearchWithVectorStore(): void {
		$store = new VectorStore( $this->dir, 8 );
		$this->seedStore( $store );

		$ivf   = new IVFIndex( $store, 5, 3 );
		$stats = $ivf->build( 'test', 4 );

		$this->assertEquals( 50, $stats['vectors'] );
		$results = $ivf->search( 'test', [1, 0, 0, 0, 0, 0, 0, 0], 5 );
		$this->assertNotEmpty( $results );
	}

	public function testBuildAndSearchWithQuantizedStore(): void {
		$store = new QuantizedStore( $this->dir, 8 );
		$this->seedStore( $store );

		$ivf   = new IVFIndex( $store, 5, 3 );
		$stats = $ivf->build( 'test', 4 );

		$this->assertEquals( 50, $stats['vectors'] );
		$results = $ivf->search( 'test', [1, 0, 0, 0, 0, 0, 0, 0], 5 );
		$this->assertNotEmpty( $results );
	}

	public function testFallbackToBruteForce(): void {
		$store = new VectorStore( $this->dir, 4 );
		$store->set( 'test', 'doc1', [1, 0, 0, 0] );

		$ivf = new IVFIndex( $store, 5, 3 );
		$results = $ivf->search( 'test', [1, 0, 0, 0], 1 );
		$this->assertCount( 1, $results );
	}

	public function testHasAndDropIndex(): void {
		$store = new VectorStore( $this->dir, 4 );
		$store->set( 'test', 'doc1', [1, 0, 0, 0] );

		$ivf = new IVFIndex( $store, 2, 1 );
		$this->assertFalse( $ivf->hasIndex( 'test' ) );

		$ivf->build( 'test', 4 );
		$this->assertTrue( $ivf->hasIndex( 'test' ) );

		$ivf->dropIndex( 'test' );
		$this->assertFalse( $ivf->hasIndex( 'test' ) );
	}

	public function testRecallVsBruteForce(): void {
		$store = new VectorStore( $this->dir, 8 );
		$this->seedStore( $store );

		$query      = [1, 0.1, 0, 0, 0, 0, 0, 0];
		$bruteForce = array_column( $store->search( 'test', $query, 5 ), 'id' );

		$ivf = new IVFIndex( $store, 5, 5 );
		$ivf->build( 'test', 8 );
		$ivfIds = array_column( $ivf->search( 'test', $query, 5 ), 'id' );

		$overlap = count( array_intersect( $bruteForce, $ivfIds ) );
		$this->assertGreaterThanOrEqual( 3, $overlap );
	}

	private function seedStore( StoreInterface $store ): void {
		for ( $i = 0; $i < 50; $i++ ) {
			$v = [];
			for ( $d = 0; $d < 8; $d++ ) {
				$v[] = sin( $i + $d ) * 0.5 + 0.5;
			}
			$store->set( 'test', "doc_$i", $v );
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
