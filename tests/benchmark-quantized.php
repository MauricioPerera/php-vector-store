<?php
/**
 * Benchmark: Float32 vs Int8 quantized — real embeddings.
 * Run: php wp-cli.phar --path=/wp eval-file benchmark-quantized.php
 */

require_once __DIR__ . '/../src/VectorStore.php';
require_once __DIR__ . '/../src/QuantizedStore.php';
require_once __DIR__ . '/../src/IVFIndex.php';

use PHPVectorStore\VectorStore;
use PHPVectorStore\QuantizedStore;
use PHPVectorStore\IVFIndex;

$dataset = array(
	array( 'id' => 'tech-1', 'text' => 'Machine learning algorithms can classify images with over 99% accuracy using convolutional neural networks.' ),
	array( 'id' => 'tech-2', 'text' => 'Kubernetes orchestrates containerized applications across clusters of machines.' ),
	array( 'id' => 'tech-3', 'text' => 'GraphQL provides a flexible query language for APIs.' ),
	array( 'id' => 'tech-4', 'text' => 'WebAssembly enables high-performance applications on the web.' ),
	array( 'id' => 'tech-5', 'text' => 'Edge computing processes data closer to where it is generated.' ),
	array( 'id' => 'sci-1', 'text' => 'CRISPR-Cas9 gene editing allows scientists to precisely modify DNA sequences.' ),
	array( 'id' => 'sci-2', 'text' => 'Black holes form when massive stars collapse at the end of their life cycle.' ),
	array( 'id' => 'sci-3', 'text' => 'Photosynthesis converts sunlight, water, and carbon dioxide into glucose.' ),
	array( 'id' => 'sci-4', 'text' => 'The human microbiome consists of trillions of bacteria that play crucial roles in health.' ),
	array( 'id' => 'biz-1', 'text' => 'Venture capital firms invest in early-stage startups with high growth potential.' ),
	array( 'id' => 'biz-2', 'text' => 'Supply chain disruptions revealed the fragility of just-in-time inventory systems.' ),
	array( 'id' => 'biz-3', 'text' => 'Remote work has fundamentally changed commercial real estate markets.' ),
	array( 'id' => 'med-1', 'text' => 'Intermittent fasting involves cycling between periods of eating and fasting.' ),
	array( 'id' => 'med-2', 'text' => 'Telemedicine allows patients to consult with healthcare providers remotely.' ),
	array( 'id' => 'med-3', 'text' => 'Antibiotic resistance occurs when bacteria evolve to survive drugs designed to kill them.' ),
	array( 'id' => 'art-1', 'text' => 'Generative AI tools create images from text descriptions, challenging artistic authorship.' ),
	array( 'id' => 'food-1', 'text' => 'Sourdough bread uses naturally occurring wild yeast and lactic acid bacteria.' ),
	array( 'id' => 'food-2', 'text' => 'The Maillard reaction creates complex flavors and brown color in grilled meats.' ),
	array( 'id' => 'sport-1', 'text' => 'Marathon running requires months of progressive training for the 42-kilometer distance.' ),
	array( 'id' => 'env-1', 'text' => 'Solar panel efficiency has increased while costs have dropped dramatically.' ),
	array( 'id' => 'edu-1', 'text' => 'Online learning platforms have democratized education with free courses from top universities.' ),
);

$queries = array(
	array( 'text' => 'How do neural networks learn?', 'expected' => 'tech-1' ),
	array( 'text' => 'What happens when a star dies?', 'expected' => 'sci-2' ),
	array( 'text' => 'Making bread at home', 'expected' => 'food-1' ),
	array( 'text' => 'Working from home trends', 'expected' => 'biz-3' ),
	array( 'text' => 'Fighting drug-resistant bacteria', 'expected' => 'med-3' ),
	array( 'text' => 'AI generating artwork', 'expected' => 'art-1' ),
	array( 'text' => 'Gene editing technology', 'expected' => 'sci-1' ),
	array( 'text' => 'Training for a marathon', 'expected' => 'sport-1' ),
	array( 'text' => 'Clean energy from sunlight', 'expected' => 'env-1' ),
	array( 'text' => 'Free university courses online', 'expected' => 'edu-1' ),
);

echo "Float32 vs Int8 Quantized — Real Embeddings\n";
echo "=============================================\n";
echo "Dataset: " . count( $dataset ) . " texts | Queries: " . count( $queries ) . "\n\n";

$dir   = sys_get_temp_dir() . '/php-vecstore-q8-' . uniqid();
$f32   = new VectorStore( $dir . '/f32', 768 );
$q8    = new QuantizedStore( $dir . '/q8', 768 );

// ── Embed all texts ─────────────────────────────────────────────────
echo "Embedding texts via Cloudflare EmbeddingGemma...\n";
$vectors = array();
foreach ( $dataset as $item ) {
	$v = cf_ai_embed( $item['text'] );
	if ( is_wp_error( $v ) ) { echo "  ERROR: " . $v->get_error_message() . "\n"; continue; }
	$vectors[ $item['id'] ] = $v;
	$f32->set( 'test', $item['id'], $v, array( 'text' => substr( $item['text'], 0, 60 ) ) );
	$q8->set( 'test', $item['id'], $v, array( 'text' => substr( $item['text'], 0, 60 ) ) );
}
$f32->flush();
$q8->flush();
echo "Done. Embedded " . count( $vectors ) . " texts.\n\n";

// Embed queries
$query_vecs = array();
foreach ( $queries as &$q ) {
	$v = cf_ai_embed( $q['text'] );
	if ( is_wp_error( $v ) ) continue;
	$q['vector'] = $v;
	$query_vecs[] = $q;
}
unset( $q );

// ── Storage comparison ──────────────────────────────────────────────
echo "=== Storage ===\n";
$f32_stats = $f32->stats();
$q8_stats  = $q8->stats();
printf( "  Float32: %d bytes (%d bytes/vec)\n", $f32_stats['total_bytes'], $f32_stats['bytes_per_vec'] );
printf( "  Int8:    %d bytes (%d bytes/vec) — %s compression\n", $q8_stats['total_bytes'], $q8_stats['bytes_per_vec'], $q8_stats['compression'] );
printf( "  Savings: %d%%\n\n", round( ( 1 - $q8_stats['total_bytes'] / max( $f32_stats['total_bytes'], 1 ) ) * 100 ) );

// ── Accuracy comparison ─────────────────────────────────────────────
echo "=== Accuracy ===\n\n";

$methods = array(
	'Float32 brute-force' => fn( $q ) => $f32->search( 'test', $q['vector'], 3 ),
	'Float32 Matryoshka'  => fn( $q ) => $f32->matryoshkaSearch( 'test', $q['vector'], 3 ),
	'Int8 brute-force'    => fn( $q ) => $q8->search( 'test', $q['vector'], 3 ),
	'Int8 Matryoshka'     => fn( $q ) => $q8->matryoshkaSearch( 'test', $q['vector'], 3 ),
);

foreach ( $methods as $name => $fn ) {
	$hits = 0;
	$top1 = 0;
	$iters = 20;
	$t = microtime( true );

	foreach ( $query_vecs as $q ) {
		$results    = $fn( $q );
		$result_ids = array_map( fn( $r ) => $r['id'], $results );

		if ( in_array( $q['expected'], $result_ids, true ) ) $hits++;
		if ( ! empty( $results ) && $results[0]['id'] === $q['expected'] ) $top1++;
	}

	// Speed: run multiple iterations
	for ( $i = 1; $i < $iters; $i++ ) {
		foreach ( $query_vecs as $q ) {
			$fn( $q );
		}
	}
	$total_ms = ( microtime( true ) - $t ) * 1000;
	$avg_ms   = round( $total_ms / ( $iters * count( $query_vecs ) ), 2 );

	$r1 = round( $top1 / count( $query_vecs ) * 100 );
	$r3 = round( $hits / count( $query_vecs ) * 100 );

	printf( "  %-25s Recall@1: %3d%%  Recall@3: %3d%%  Avg: %.2fms/q\n", $name, $r1, $r3, $avg_ms );
}

echo "\n=== Score Comparison (query: 'How do neural networks learn?') ===\n";
$q_vec     = $query_vecs[0]['vector'];
$f32_res   = $f32->search( 'test', $q_vec, 5 );
$q8_res    = $q8->search( 'test', $q_vec, 5 );

printf( "  %-12s %-15s %-15s %-10s\n", 'Rank', 'Float32', 'Int8', 'Drift' );
for ( $i = 0; $i < min( 5, count( $f32_res ) ); $i++ ) {
	$f_score = $f32_res[ $i ]['score'];
	$q_score = $q8_res[ $i ]['score'] ?? 0;
	$drift   = abs( $f_score - $q_score );
	$same_id = ( $f32_res[ $i ]['id'] ?? '' ) === ( $q8_res[ $i ]['id'] ?? '' );
	printf( "  #%-11d %-8s %.4f  %-8s %.4f  %.4f %s\n",
		$i + 1,
		$f32_res[ $i ]['id'], $f_score,
		$q8_res[ $i ]['id'] ?? '?', $q_score,
		$drift,
		$same_id ? '' : ' ← RANK CHANGE'
	);
}

// Cleanup
$f32->drop( 'test' );
$q8->drop( 'test' );
array_map( 'unlink', array_merge( glob( "$dir/f32/*" ), glob( "$dir/q8/*" ) ) );
@rmdir( "$dir/f32" );
@rmdir( "$dir/q8" );
@rmdir( $dir );

echo "\nDone.\n";
