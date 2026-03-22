<?php
/**
 * Benchmark: IVF vs brute-force vs Matryoshka at scale.
 * Run: php -d memory_limit=1G tests/benchmark-ivf.php
 */

require_once __DIR__ . '/../src/VectorStore.php';
require_once __DIR__ . '/../src/IVFIndex.php';

use PHPVectorStore\VectorStore;
use PHPVectorStore\IVFIndex;

$dir = sys_get_temp_dir() . '/php-vecstore-ivf-' . uniqid();
echo "IVF Index Benchmark\n";
echo "====================\n";
echo "PHP: " . PHP_VERSION . "\n\n";

function random_vector( int $dim = 768 ): array {
	$v = array();
	for ( $i = 0; $i < $dim; $i++ ) {
		$v[] = ( mt_rand() / mt_getrandmax() ) * 2 - 1;
	}
	return $v;
}

$sizes = array( 1000, 5000, 10000 );

foreach ( $sizes as $n ) {
	echo "=== $n vectors ===\n";

	$store = new VectorStore( $dir . "/$n", 768 );

	// Insert vectors
	$t = microtime( true );
	for ( $i = 0; $i < $n; $i++ ) {
		$store->set( 'test', "v_$i", random_vector(), array( 'i' => $i ) );
	}
	$store->flush();
	$insert_ms = round( ( microtime( true ) - $t ) * 1000 );
	echo "Insert: {$insert_ms}ms\n";

	$query = random_vector();
	$iters = max( 5, min( 50, (int) ( 5000 / $n ) ) );

	// Brute-force 768d
	$t = microtime( true );
	for ( $i = 0; $i < $iters; $i++ ) {
		$bf_results = $store->search( 'test', $query, 5 );
	}
	$bf_ms = round( ( microtime( true ) - $t ) / $iters * 1000, 1 );

	// Matryoshka 3-stage (no IVF)
	$t = microtime( true );
	for ( $i = 0; $i < $iters; $i++ ) {
		$mat_results = $store->matryoshkaSearch( 'test', $query, 5, [128, 384, 768] );
	}
	$mat_ms = round( ( microtime( true ) - $t ) / $iters * 1000, 1 );

	// Build IVF index
	$k      = max( 10, (int) ceil( sqrt( $n ) ) );
	$probes = max( 5, (int) ceil( $k * 0.2 ) ); // 20% probes for better recall
	$ivf    = new IVFIndex( $store, $k, $probes );

	$t          = microtime( true );
	$build_info = $ivf->build( 'test', 128 );
	$build_ms   = round( ( microtime( true ) - $t ) * 1000 );

	echo "IVF build: {$build_ms}ms (K=$k, avg_cluster={$build_info['avg_cluster']})\n";

	// IVF search
	$t = microtime( true );
	for ( $i = 0; $i < $iters; $i++ ) {
		$ivf_results = $ivf->search( 'test', $query, 5 );
	}
	$ivf_ms = round( ( microtime( true ) - $t ) / $iters * 1000, 1 );

	// IVF + Matryoshka
	$t = microtime( true );
	for ( $i = 0; $i < $iters; $i++ ) {
		$ivfm_results = $ivf->matryoshkaSearch( 'test', $query, 5, [128, 384, 768] );
	}
	$ivfm_ms = round( ( microtime( true ) - $t ) / $iters * 1000, 1 );

	// Accuracy: compare IVF top-5 IDs vs brute-force top-5
	$bf_ids   = array_map( fn( $r ) => $r['id'], $bf_results );
	$ivf_ids  = array_map( fn( $r ) => $r['id'], $ivf_results );
	$ivfm_ids = array_map( fn( $r ) => $r['id'], $ivfm_results );
	$recall_ivf  = count( array_intersect( $bf_ids, $ivf_ids ) ) / count( $bf_ids ) * 100;
	$recall_ivfm = count( array_intersect( $bf_ids, $ivfm_ids ) ) / count( $bf_ids ) * 100;

	echo "\n";
	printf( "  %-30s %8s %8s %8s\n", 'Method', 'Time', 'Speedup', 'Recall' );
	printf( "  %-30s %8s %8s %8s\n", '------', '----', '-------', '------' );
	printf( "  %-30s %7.1fms %7s %7s\n", 'Brute-force 768d', $bf_ms, '1.0x', '100%' );
	printf( "  %-30s %7.1fms %6.1fx %7s\n", 'Matryoshka 128→384→768', $mat_ms, $bf_ms / max( $mat_ms, 0.1 ), '~100%' );
	printf( "  %-30s %7.1fms %6.1fx %5.0f%%\n", 'IVF (K=' . $k . ')', $ivf_ms, $bf_ms / max( $ivf_ms, 0.1 ), $recall_ivf );
	printf( "  %-30s %7.1fms %6.1fx %5.0f%%\n", 'IVF + Matryoshka', $ivfm_ms, $bf_ms / max( $ivfm_ms, 0.1 ), $recall_ivfm );
	echo "\n";

	// Cleanup
	$store->drop( 'test' );
	$ivf->dropIndex( 'test' );
	array_map( 'unlink', glob( $dir . "/$n/*" ) );
	@rmdir( $dir . "/$n" );
}

echo "Peak memory: " . round( memory_get_peak_usage( true ) / ( 1024 * 1024 ), 1 ) . " MB\n";

@rmdir( $dir );
echo "Done.\n";
