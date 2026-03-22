<?php
/**
 * Benchmark: PHP Vector Store vs sqlite-vec equivalent operations.
 *
 * Tests: insert, search, matryoshka search at various dataset sizes.
 * Run: php tests/benchmark.php
 */

require_once __DIR__ . '/../src/VectorStore.php';

use PHPVectorStore\VectorStore;

$dir = sys_get_temp_dir() . '/php-vecstore-bench-' . uniqid();
echo "PHP Vector Store Benchmark\n";
echo "==========================\n";
echo "Directory: $dir\n";
echo "PHP: " . PHP_VERSION . "\n\n";

$store = new VectorStore( $dir, 768 );

// Generate random vectors
function random_vector( int $dim = 768 ): array {
	$v = array();
	for ( $i = 0; $i < $dim; $i++ ) {
		$v[] = ( mt_rand() / mt_getrandmax() ) * 2 - 1;
	}
	return $v;
}

$sizes = array( 100, 1000, 5000 );

foreach ( $sizes as $n ) {
	echo "--- $n vectors ---\n";

	// Pre-generate vectors
	$vectors = array();
	for ( $i = 0; $i < $n; $i++ ) {
		$vectors[] = random_vector();
	}

	// Insert
	$t = microtime( true );
	for ( $i = 0; $i < $n; $i++ ) {
		$store->set( "bench_$n", "vec_$i", $vectors[ $i ], array( 'index' => $i ) );
	}
	$store->flush();
	$insert_ms = round( ( microtime( true ) - $t ) * 1000, 1 );
	echo "  Insert: {$insert_ms}ms (" . round( $n / ( $insert_ms / 1000 ) ) . " vec/s)\n";

	// Storage size
	$stats = $store->stats();
	$coll  = null;
	foreach ( $stats['collections'] as $c ) {
		if ( $c['collection'] === "bench_$n" ) {
			$coll = $c;
			break;
		}
	}
	$size_kb = round( $coll['bytes'] / 1024, 1 );
	echo "  Size: {$size_kb} KB ({$coll['vectors']} vectors)\n";

	// Search (full 768d)
	$query = random_vector();
	$t     = microtime( true );
	$iters = min( 100, max( 10, (int) ( 10000 / $n ) ) );
	for ( $i = 0; $i < $iters; $i++ ) {
		$store->search( "bench_$n", $query, 5 );
	}
	$search_ms = round( ( microtime( true ) - $t ) / $iters * 1000, 2 );
	echo "  Search 768d: {$search_ms}ms/query\n";

	// Matryoshka search (128d → 768d)
	$t = microtime( true );
	for ( $i = 0; $i < $iters; $i++ ) {
		$store->matryoshkaSearch( "bench_$n", $query, 5, 128, 3 );
	}
	$mat_ms = round( ( microtime( true ) - $t ) / $iters * 1000, 2 );
	$speedup = $search_ms > 0 ? round( $search_ms / $mat_ms, 1 ) : 0;
	echo "  Matryoshka 128→768: {$mat_ms}ms/query ({$speedup}x faster)\n";

	// Search (128d only)
	$t = microtime( true );
	for ( $i = 0; $i < $iters; $i++ ) {
		$store->search( "bench_$n", $query, 5, 128 );
	}
	$coarse_ms = round( ( microtime( true ) - $t ) / $iters * 1000, 2 );
	echo "  Search 128d only: {$coarse_ms}ms/query\n";

	// Cleanup
	$store->drop( "bench_$n" );
	echo "\n";
}

// Memory usage
echo "Peak memory: " . round( memory_get_peak_usage( true ) / ( 1024 * 1024 ), 1 ) . " MB\n";

// Comparison table
echo "\n=== vs sqlite-vec comparison ===\n";
echo "Feature                    | PHP Vector Store    | sqlite-vec\n";
echo "---------------------------|---------------------|-------------------\n";
echo "Dependencies               | None (pure PHP)     | C extension + FFI\n";
echo "Storage format             | Float32 binary      | SQLite virtual table\n";
echo "Size per 768d vector       | 3,072 bytes         | ~3,100 bytes\n";
echo "Search algorithm           | Brute-force + Matry | ANN (IVF/HNSW)\n";
echo "Best for                   | <50K vectors        | >50K vectors\n";
echo "Matryoshka support         | Native (dimSlice)   | Manual\n";
echo "Crash safety               | Atomic rename       | SQLite WAL\n";
echo "Multi-collection           | Per-file scoping    | Per-table\n";
echo "Metadata support           | JSON sidecar        | Extra columns\n";
echo "WordPress integration      | Drop-in             | Requires extension\n";
echo "PHP version                | 8.1+                | 8.1+ with FFI\n";

// Cleanup temp directory
array_map( 'unlink', glob( "$dir/*" ) );
rmdir( $dir );
echo "\nDone.\n";
