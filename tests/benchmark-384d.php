<?php
/**
 * Benchmark: 768d vs 384d — accuracy and speed with real embeddings.
 * Run: php wp-cli.phar --path=/wp eval-file benchmark-384d.php
 */

require_once __DIR__ . '/../src/VectorStore.php';
require_once __DIR__ . '/../src/QuantizedStore.php';

use PHPVectorStore\VectorStore;
use PHPVectorStore\QuantizedStore;

$dataset = array(
	array( 'id' => 'tech-1', 'text' => 'Machine learning algorithms can classify images with over 99% accuracy using convolutional neural networks.' ),
	array( 'id' => 'tech-2', 'text' => 'Kubernetes orchestrates containerized applications across clusters of machines.' ),
	array( 'id' => 'tech-3', 'text' => 'GraphQL provides a flexible query language for APIs.' ),
	array( 'id' => 'tech-4', 'text' => 'WebAssembly enables high-performance applications on the web.' ),
	array( 'id' => 'tech-5', 'text' => 'Edge computing processes data closer to where it is generated.' ),
	array( 'id' => 'sci-1', 'text' => 'CRISPR-Cas9 gene editing allows scientists to precisely modify DNA sequences.' ),
	array( 'id' => 'sci-2', 'text' => 'Black holes form when massive stars collapse at the end of their life cycle.' ),
	array( 'id' => 'sci-3', 'text' => 'Photosynthesis converts sunlight, water, and carbon dioxide into glucose.' ),
	array( 'id' => 'sci-4', 'text' => 'The human microbiome consists of trillions of bacteria in the body.' ),
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

echo "768d vs 384d — Real Embedding Comparison\n";
echo "==========================================\n\n";

$dir = sys_get_temp_dir() . '/php-vecstore-dim-' . uniqid();

// 4 stores: f32@768, f32@384, q8@768, q8@384
$f32_768 = new VectorStore( $dir . '/f32-768', 768 );
$f32_384 = new VectorStore( $dir . '/f32-384', 384 );
$q8_768  = new QuantizedStore( $dir . '/q8-768', 768 );
$q8_384  = new QuantizedStore( $dir . '/q8-384', 384 );

// Embed
echo "Embedding " . count( $dataset ) . " texts...\n";
foreach ( $dataset as $item ) {
	$v = cf_ai_embed( $item['text'] );
	if ( is_wp_error( $v ) ) continue;

	$v384 = array_slice( $v, 0, 384 );

	$f32_768->set( 't', $item['id'], $v );
	$f32_384->set( 't', $item['id'], $v384 );
	$q8_768->set( 't', $item['id'], $v );
	$q8_384->set( 't', $item['id'], $v384 );
}
$f32_768->flush(); $f32_384->flush(); $q8_768->flush(); $q8_384->flush();
echo "Done.\n\n";

// Embed queries
foreach ( $queries as &$q ) {
	$v = cf_ai_embed( $q['text'] );
	if ( is_wp_error( $v ) ) continue;
	$q['v768'] = $v;
	$q['v384'] = array_slice( $v, 0, 384 );
}
unset( $q );

// ── Storage ─────────────────────────────────────────────────────────
echo "=== Storage per vector ===\n";
printf( "  Float32 @ 768d: %d bytes\n", $f32_768->stats()['bytes_per_vec'] );
printf( "  Float32 @ 384d: %d bytes\n", $f32_384->stats()['bytes_per_vec'] );
printf( "  Int8    @ 768d: %d bytes\n", $q8_768->stats()['bytes_per_vec'] );
printf( "  Int8    @ 384d: %d bytes\n\n", $q8_384->stats()['bytes_per_vec'] );

// ── Accuracy + Speed ────────────────────────────────────────────────
echo "=== Accuracy & Speed ===\n\n";

$configs = array(
	'Float32 768d brute'       => fn( $q ) => $f32_768->search( 't', $q['v768'], 3 ),
	'Float32 768d Matryoshka'  => fn( $q ) => $f32_768->matryoshkaSearch( 't', $q['v768'], 3, [128, 384, 768] ),
	'Float32 384d brute'       => fn( $q ) => $f32_384->search( 't', $q['v384'], 3 ),
	'Float32 384d Matryoshka'  => fn( $q ) => $f32_384->matryoshkaSearch( 't', $q['v384'], 3, [128, 256, 384] ),
	'Int8 768d brute'          => fn( $q ) => $q8_768->search( 't', $q['v768'], 3 ),
	'Int8 768d Matryoshka'     => fn( $q ) => $q8_768->matryoshkaSearch( 't', $q['v768'], 3, [128, 384, 768] ),
	'Int8 384d brute'          => fn( $q ) => $q8_384->search( 't', $q['v384'], 3 ),
	'Int8 384d Matryoshka'     => fn( $q ) => $q8_384->matryoshkaSearch( 't', $q['v384'], 3, [128, 256, 384] ),
);

$iters = 20;

printf( "  %-30s %8s %8s %10s\n", 'Config', 'R@1', 'R@3', 'Avg ms/q' );
printf( "  %-30s %8s %8s %10s\n", '------', '---', '---', '--------' );

foreach ( $configs as $name => $fn ) {
	$top1 = 0; $top3 = 0;

	$t = microtime( true );
	for ( $iter = 0; $iter < $iters; $iter++ ) {
		foreach ( $queries as $q ) {
			if ( ! isset( $q['v768'] ) ) continue;
			$results = $fn( $q );
			if ( $iter === 0 ) {
				$ids = array_map( fn( $r ) => $r['id'], $results );
				if ( ! empty( $results ) && $results[0]['id'] === $q['expected'] ) $top1++;
				if ( in_array( $q['expected'], $ids, true ) ) $top3++;
			}
		}
	}
	$avg = round( ( microtime( true ) - $t ) / ( $iters * count( $queries ) ) * 1000, 2 );
	$r1  = round( $top1 / count( $queries ) * 100 );
	$r3  = round( $top3 / count( $queries ) * 100 );

	printf( "  %-30s %7d%% %7d%% %9sms\n", $name, $r1, $r3, $avg );
}

// ── Score comparison ────────────────────────────────────────────────
echo "\n=== Score comparison: 'How do neural networks learn?' ===\n";
$r768 = $f32_768->search( 't', $queries[0]['v768'], 5 );
$r384 = $f32_384->search( 't', $queries[0]['v384'], 5 );

printf( "  %-6s %-10s %-12s %-10s %-12s\n", 'Rank', 'ID@768', 'Score@768', 'ID@384', 'Score@384' );
for ( $i = 0; $i < 5; $i++ ) {
	$same = ( $r768[$i]['id'] ?? '' ) === ( $r384[$i]['id'] ?? '' ) ? '' : ' *';
	printf( "  #%-5d %-10s %-12.4f %-10s %-12.4f%s\n",
		$i + 1,
		$r768[$i]['id'] ?? '?', $r768[$i]['score'] ?? 0,
		$r384[$i]['id'] ?? '?', $r384[$i]['score'] ?? 0,
		$same
	);
}

// Cleanup
foreach ( array( 'f32-768', 'f32-384', 'q8-768', 'q8-384' ) as $sub ) {
	array_map( 'unlink', glob( "$dir/$sub/*" ) );
	@rmdir( "$dir/$sub" );
}
@rmdir( $dir );

echo "\nDone.\n";
