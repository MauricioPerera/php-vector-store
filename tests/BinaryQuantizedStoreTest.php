<?php

namespace PHPVectorStore\Tests;

use PHPUnit\Framework\TestCase;
use PHPVectorStore\BinaryQuantizedStore;
use PHPVectorStore\StoreInterface;
use PHPVectorStore\Distance;
use PHPVectorStore\Exception\DimensionMismatchException;

class BinaryQuantizedStoreTest extends TestCase
{
	private string $dir;

	protected function setUp(): void {
		$this->dir = sys_get_temp_dir() . '/phpvs_b1test_' . uniqid();
		mkdir( $this->dir, 0755, true );
	}

	protected function tearDown(): void {
		$this->removeDir( $this->dir );
	}

	public function testImplementsStoreInterface(): void {
		$this->assertInstanceOf( StoreInterface::class, new BinaryQuantizedStore( $this->dir, 4 ) );
	}

	public function testBytesPerVector(): void {
		$store4   = new BinaryQuantizedStore( $this->dir, 4 );
		$store768 = new BinaryQuantizedStore( $this->dir, 768 );

		$this->assertEquals( 1, $store4->bytesPerVector() );     // ceil(4/8) = 1
		$this->assertEquals( 96, $store768->bytesPerVector() );  // ceil(768/8) = 96
	}

	public function testSetAndGet(): void {
		$store = new BinaryQuantizedStore( $this->dir, 4 );
		$store->set( 'test', 'doc1', [1.0, 0.0, 0.0, 0.0], ['key' => 'val'] );

		$this->assertTrue( $store->has( 'test', 'doc1' ) );
		$result = $store->get( 'test', 'doc1' );
		$this->assertNotNull( $result );
		$this->assertEquals( ['key' => 'val'], $result['metadata'] );
	}

	public function testBinaryQuantizationPreservesSign(): void {
		$original  = [0.5, -0.3, 0.8, -0.1];
		$quantized = BinaryQuantizedStore::quantize( $original, 4 );
		$restored  = BinaryQuantizedStore::dequantize( $quantized, 0, 4 );

		// Binary quantization preserves sign: positive → +1.0, negative → -1.0
		$this->assertEquals( 1.0, $restored[0] );   // 0.5 → +1.0
		$this->assertEquals( -1.0, $restored[1] );   // -0.3 → -1.0
		$this->assertEquals( 1.0, $restored[2] );   // 0.8 → +1.0
		$this->assertEquals( -1.0, $restored[3] );   // -0.1 → -1.0
	}

	public function testSearchRanking(): void {
		$store = new BinaryQuantizedStore( $this->dir, 8 );
		// Binary quantization only keeps sign — need clearly different sign patterns
		$store->set( 'test', 'close',  [1.0, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5] );  // signs: ++++ ----
		$store->set( 'test', 'far',    [-1.0, -0.5, -0.3, -0.1, 0.1, 0.2, 0.3, 0.5] );   // signs: ---- ++++  (opposite)
		$store->set( 'test', 'medium', [1.0, 0.5, 0.3, -0.1, -0.1, -0.2, 0.3, -0.5] );   // signs: +++- --+-  (6/8 match)

		$results = $store->search( 'test', [1.0, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5], 3 );
		$this->assertGreaterThanOrEqual( 1, count( $results ) );
		$this->assertEquals( 'close', $results[0]['id'] );
	}

	public function testSearchWithDistance(): void {
		$store = new BinaryQuantizedStore( $this->dir, 8 );
		$store->set( 'test', 'close', [1.0, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5] );
		$store->set( 'test', 'far',   [-1.0, -0.5, -0.3, -0.1, 0.1, 0.2, 0.3, 0.5] );

		$results = $store->search( 'test', [1.0, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5], 2, 0, Distance::Euclidean );
		$this->assertEquals( 'close', $results[0]['id'] );
	}

	public function testSearchAcross(): void {
		$store = new BinaryQuantizedStore( $this->dir, 8 );
		$store->set( 'col_a', 'a1', [1, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5] );
		$store->set( 'col_b', 'b1', [0.9, 0.4, 0.2, 0.1, -0.1, -0.3, -0.2, -0.4] );

		$results = $store->searchAcross( ['col_a', 'col_b'], [1, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5] );
		$this->assertGreaterThanOrEqual( 2, count( $results ) );
	}

	public function testMatryoshkaSearch(): void {
		$store = new BinaryQuantizedStore( $this->dir, 16 );
		$store->set( 'test', 'best',  [1, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5, 1, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5] );
		$store->set( 'test', 'worst', [-1, -0.5, -0.3, -0.1, 0.1, 0.2, 0.3, 0.5, -1, -0.5, -0.3, -0.1, 0.1, 0.2, 0.3, 0.5] );
		$store->set( 'test', 'mid',   [1, 0.5, -0.3, -0.1, 0.1, -0.2, 0.3, -0.5, 1, -0.5, 0.3, -0.1, 0.1, -0.2, 0.3, -0.5] );

		$results = $store->matryoshkaSearch( 'test', [1, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5, 1, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5], 2, [8, 16] );
		$this->assertEquals( 'best', $results[0]['id'] );
		$this->assertArrayHasKey( 'stages', $results[0] );
	}

	public function testImportExport(): void {
		$store = new BinaryQuantizedStore( $this->dir, 4 );
		$records = [
			['id' => 'r1', 'vector' => [1, 0, 0, 0]],
			['id' => 'r2', 'vector' => [0, 1, 0, 0]],
		];
		$this->assertEquals( 2, $store->import( 'test', $records ) );
		$this->assertCount( 2, $store->export( 'test' ) );
	}

	public function testDimensionMismatchThrows(): void {
		$store = new BinaryQuantizedStore( $this->dir, 4 );
		$this->expectException( DimensionMismatchException::class );
		$store->set( 'test', 'doc1', [1.0, 0.0] ); // 2 dims, expects 4
	}

	public function testFlushAndReload(): void {
		$store = new BinaryQuantizedStore( $this->dir, 8 );
		$store->set( 'test', 'doc1', [1, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3, -0.5], ['x' => 1] );
		$store->flush();

		$store2 = new BinaryQuantizedStore( $this->dir, 8 );
		$result = $store2->get( 'test', 'doc1' );
		$this->assertNotNull( $result );
		$this->assertEquals( ['x' => 1], $result['metadata'] );
	}

	public function testRemove(): void {
		$store = new BinaryQuantizedStore( $this->dir, 4 );
		$store->set( 'test', 'doc1', [1, 0, 0, 0] );
		$store->set( 'test', 'doc2', [0, 1, 0, 0] );

		$this->assertTrue( $store->remove( 'test', 'doc1' ) );
		$this->assertFalse( $store->has( 'test', 'doc1' ) );
		$this->assertTrue( $store->has( 'test', 'doc2' ) );
		$this->assertEquals( 1, $store->count( 'test' ) );
	}

	public function testStats(): void {
		$store = new BinaryQuantizedStore( $this->dir, 768 );
		$store->set( 'test', 'doc1', array_fill( 0, 768, 0.5 ) );
		$store->flush();

		$stats = $store->stats();
		$this->assertEquals( 'binary1bit', $stats['quantization'] );
		$this->assertEquals( 96, $stats['bytes_per_vec'] );
		$this->assertEquals( 1, $stats['total_vectors'] );
		$this->assertEquals( '32x', $stats['compression'] );
	}

	public function testHammingAccuracy(): void {
		// Two identical normalized vectors should have cosine sim ≈ 1.0
		$v = [0.5, -0.3, 0.8, -0.1, 0.6, -0.7, 0.2, -0.4];
		$q = BinaryQuantizedStore::quantize( $v, 8 );

		// Same vector XOR itself = 0 hamming = cosine 1.0
		$store = new BinaryQuantizedStore( $this->dir, 8 );
		$store->set( 'test', 'same', $v );

		$results = $store->search( 'test', $v, 1 );
		$this->assertCount( 1, $results );
		$this->assertEqualsWithDelta( 1.0, $results[0]['score'], 0.01 );
	}

	private function removeDir( string $dir ): void {
		if ( ! is_dir( $dir ) ) return;
		foreach ( glob( $dir . '/*' ) ?: [] as $file ) {
			is_dir( $file ) ? $this->removeDir( $file ) : unlink( $file );
		}
		rmdir( $dir );
	}
}
