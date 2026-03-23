<?php

namespace PHPVectorStore\Tests;

use PHPUnit\Framework\TestCase;
use PHPVectorStore\QuantizedStore;
use PHPVectorStore\StoreInterface;
use PHPVectorStore\Distance;
use PHPVectorStore\Exception\DimensionMismatchException;

class QuantizedStoreTest extends TestCase
{
	private string $dir;

	protected function setUp(): void {
		$this->dir = sys_get_temp_dir() . '/phpvs_qtest_' . uniqid();
		mkdir( $this->dir, 0755, true );
	}

	protected function tearDown(): void {
		$this->removeDir( $this->dir );
	}

	public function testImplementsStoreInterface(): void {
		$this->assertInstanceOf( StoreInterface::class, new QuantizedStore( $this->dir, 4 ) );
	}

	public function testSetAndGet(): void {
		$store = new QuantizedStore( $this->dir, 4 );
		$store->set( 'test', 'doc1', [1.0, 0.0, 0.0, 0.0], ['key' => 'val'] );

		$this->assertTrue( $store->has( 'test', 'doc1' ) );
		$result = $store->get( 'test', 'doc1' );
		$this->assertNotNull( $result );
		$this->assertEquals( ['key' => 'val'], $result['metadata'] );
	}

	public function testQuantizationRoundTrip(): void {
		$original  = [0.5, -0.3, 0.8, -0.1];
		$quantized = QuantizedStore::quantize( $original, 4 );
		$restored  = QuantizedStore::dequantize( $quantized, 0, 4 );

		for ( $i = 0; $i < 4; $i++ ) {
			$this->assertEqualsWithDelta( $original[ $i ], $restored[ $i ], 0.02, "Dim $i" );
		}
	}

	public function testSearchRanking(): void {
		$store = new QuantizedStore( $this->dir, 4 );
		$store->set( 'test', 'close',  [1.0, 0.1, 0.0, 0.0] );
		$store->set( 'test', 'far',    [0.0, 0.0, 0.0, 1.0] );
		$store->set( 'test', 'medium', [0.5, 0.5, 0.0, 0.0] );

		$results = $store->search( 'test', [1.0, 0.0, 0.0, 0.0], 3 );
		$this->assertGreaterThanOrEqual( 2, count( $results ) );
		$this->assertEquals( 'close', $results[0]['id'] );
	}

	public function testSearchWithDistance(): void {
		$store = new QuantizedStore( $this->dir, 4 );
		$store->set( 'test', 'close', [1.0, 0.0, 0.0, 0.0] );
		$store->set( 'test', 'far',   [0.0, 0.0, 0.0, 1.0] );

		$results = $store->search( 'test', [1.0, 0.0, 0.0, 0.0], 2, 0, Distance::Euclidean );
		$this->assertEquals( 'close', $results[0]['id'] );
	}

	public function testSearchAcross(): void {
		$store = new QuantizedStore( $this->dir, 4 );
		$store->set( 'col_a', 'a1', [1, 0, 0, 0] );
		$store->set( 'col_b', 'b1', [0.9, 0.1, 0, 0] );

		$results = $store->searchAcross( ['col_a', 'col_b'], [1, 0, 0, 0] );
		$this->assertGreaterThanOrEqual( 2, count( $results ) );
	}

	public function testImportExport(): void {
		$store = new QuantizedStore( $this->dir, 4 );
		$records = [
			['id' => 'r1', 'vector' => [1, 0, 0, 0]],
			['id' => 'r2', 'vector' => [0, 1, 0, 0]],
		];
		$this->assertEquals( 2, $store->import( 'test', $records ) );
		$this->assertCount( 2, $store->export( 'test' ) );
	}

	public function testDimensionMismatchThrows(): void {
		$store = new QuantizedStore( $this->dir, 4 );
		$this->expectException( DimensionMismatchException::class );
		$store->set( 'test', 'doc1', [1.0, 0.0] ); // 2 dims, expects 4
	}

	public function testFlushAndReload(): void {
		$store = new QuantizedStore( $this->dir, 4 );
		$store->set( 'test', 'doc1', [1, 0, 0, 0], ['x' => 1] );
		$store->flush();

		$store2 = new QuantizedStore( $this->dir, 4 );
		$result = $store2->get( 'test', 'doc1' );
		$this->assertNotNull( $result );
		$this->assertEquals( ['x' => 1], $result['metadata'] );
	}

	private function removeDir( string $dir ): void {
		if ( ! is_dir( $dir ) ) return;
		foreach ( glob( $dir . '/*' ) ?: [] as $file ) {
			is_dir( $file ) ? $this->removeDir( $file ) : unlink( $file );
		}
		rmdir( $dir );
	}
}
