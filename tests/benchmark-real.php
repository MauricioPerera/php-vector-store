<?php
/**
 * Real-world benchmark using Cloudflare Workers AI EmbeddingGemma-300m.
 *
 * Generates embeddings from a diverse text dataset, stores them,
 * and benchmarks search accuracy + speed with real semantic queries.
 *
 * Run from WP context:
 *   php wp-cli.phar --path=/path/to/wp eval-file /path/to/benchmark-real.php
 */

require_once __DIR__ . '/../src/VectorStore.php';
require_once __DIR__ . '/../src/IVFIndex.php';

use PHPVectorStore\VectorStore;
use PHPVectorStore\IVFIndex;

// ── Dataset: diverse topics for real semantic structure ──────────────
$dataset = array(
	// Technology
	array( 'id' => 'tech-1', 'text' => 'Machine learning algorithms can classify images with over 99% accuracy using convolutional neural networks trained on millions of labeled examples.', 'topic' => 'technology' ),
	array( 'id' => 'tech-2', 'text' => 'Kubernetes orchestrates containerized applications across clusters of machines, providing automated deployment, scaling, and management of workloads.', 'topic' => 'technology' ),
	array( 'id' => 'tech-3', 'text' => 'GraphQL provides a flexible query language for APIs that allows clients to request exactly the data they need, reducing over-fetching and under-fetching.', 'topic' => 'technology' ),
	array( 'id' => 'tech-4', 'text' => 'WebAssembly enables high-performance applications on the web by compiling languages like C++ and Rust to run at near-native speed in browsers.', 'topic' => 'technology' ),
	array( 'id' => 'tech-5', 'text' => 'Edge computing processes data closer to where it is generated rather than sending everything to centralized cloud data centers.', 'topic' => 'technology' ),
	array( 'id' => 'tech-6', 'text' => 'Rust programming language provides memory safety without garbage collection through its ownership and borrowing system.', 'topic' => 'technology' ),
	array( 'id' => 'tech-7', 'text' => 'Quantum computers use qubits that can exist in superposition, enabling them to solve certain problems exponentially faster than classical computers.', 'topic' => 'technology' ),
	array( 'id' => 'tech-8', 'text' => 'Blockchain technology creates an immutable distributed ledger that records transactions across many computers without a central authority.', 'topic' => 'technology' ),

	// Science
	array( 'id' => 'sci-1', 'text' => 'CRISPR-Cas9 gene editing allows scientists to precisely modify DNA sequences in living organisms, opening possibilities for treating genetic diseases.', 'topic' => 'science' ),
	array( 'id' => 'sci-2', 'text' => 'Black holes form when massive stars collapse at the end of their life cycle, creating regions where gravity is so strong that nothing can escape.', 'topic' => 'science' ),
	array( 'id' => 'sci-3', 'text' => 'Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen, providing the foundation for most life on Earth.', 'topic' => 'science' ),
	array( 'id' => 'sci-4', 'text' => 'The human microbiome consists of trillions of bacteria that play crucial roles in digestion, immunity, and even mental health.', 'topic' => 'science' ),
	array( 'id' => 'sci-5', 'text' => 'Plate tectonics explains how the Earth surface is divided into moving plates that cause earthquakes, volcanoes, and mountain formation.', 'topic' => 'science' ),
	array( 'id' => 'sci-6', 'text' => 'Neutrinos are subatomic particles with almost no mass that pass through ordinary matter almost undisturbed, making them extremely difficult to detect.', 'topic' => 'science' ),
	array( 'id' => 'sci-7', 'text' => 'Climate change is driven primarily by greenhouse gas emissions from burning fossil fuels, causing global temperatures to rise and weather patterns to shift.', 'topic' => 'science' ),
	array( 'id' => 'sci-8', 'text' => 'RNA vaccines work by instructing cells to produce a harmless piece of the target virus protein, training the immune system to recognize and fight the real pathogen.', 'topic' => 'science' ),

	// Business & Finance
	array( 'id' => 'biz-1', 'text' => 'Venture capital firms invest in early-stage startups with high growth potential in exchange for equity ownership and board representation.', 'topic' => 'business' ),
	array( 'id' => 'biz-2', 'text' => 'Supply chain disruptions caused by the pandemic revealed the fragility of just-in-time inventory systems used by manufacturers worldwide.', 'topic' => 'business' ),
	array( 'id' => 'biz-3', 'text' => 'Remote work has fundamentally changed commercial real estate markets as companies reduce office space and adopt hybrid work models.', 'topic' => 'business' ),
	array( 'id' => 'biz-4', 'text' => 'Central banks use interest rate adjustments as a primary tool to control inflation and stimulate or cool down economic activity.', 'topic' => 'business' ),
	array( 'id' => 'biz-5', 'text' => 'E-commerce platforms have transformed retail by enabling small businesses to reach global customers without physical storefronts.', 'topic' => 'business' ),
	array( 'id' => 'biz-6', 'text' => 'Cryptocurrency exchanges facilitate trading of digital assets like Bitcoin and Ethereum, operating as the stock markets of the crypto world.', 'topic' => 'business' ),

	// Health & Medicine
	array( 'id' => 'med-1', 'text' => 'Intermittent fasting involves cycling between periods of eating and fasting, which may improve metabolic health and promote cellular repair.', 'topic' => 'health' ),
	array( 'id' => 'med-2', 'text' => 'Telemedicine allows patients to consult with healthcare providers remotely using video calls, expanding access to medical care in rural areas.', 'topic' => 'health' ),
	array( 'id' => 'med-3', 'text' => 'Antibiotic resistance occurs when bacteria evolve to survive the drugs designed to kill them, posing a serious threat to global public health.', 'topic' => 'health' ),
	array( 'id' => 'med-4', 'text' => 'Meditation and mindfulness practices have been shown to reduce stress, anxiety, and depression while improving focus and emotional regulation.', 'topic' => 'health' ),

	// Arts & Culture
	array( 'id' => 'art-1', 'text' => 'Generative AI tools like DALL-E and Midjourney create images from text descriptions, challenging traditional notions of artistic authorship and creativity.', 'topic' => 'arts' ),
	array( 'id' => 'art-2', 'text' => 'Streaming platforms have disrupted the music industry by providing instant access to millions of songs while changing how artists earn revenue.', 'topic' => 'arts' ),
	array( 'id' => 'art-3', 'text' => 'Ancient Greek philosophy laid the foundations for Western thought, with Socrates, Plato, and Aristotle exploring ethics, politics, and metaphysics.', 'topic' => 'arts' ),
	array( 'id' => 'art-4', 'text' => 'The Renaissance was a cultural movement that began in Italy in the 14th century, marked by renewed interest in classical art, science, and humanism.', 'topic' => 'arts' ),

	// Food & Cooking
	array( 'id' => 'food-1', 'text' => 'Sourdough bread uses naturally occurring wild yeast and lactic acid bacteria for fermentation, producing a distinctive tangy flavor and chewy texture.', 'topic' => 'food' ),
	array( 'id' => 'food-2', 'text' => 'The Maillard reaction occurs when amino acids and sugars are heated together, creating complex flavors and brown color in grilled meats and toasted bread.', 'topic' => 'food' ),
	array( 'id' => 'food-3', 'text' => 'Japanese sushi originated as a method of preserving fish by fermenting it with rice, evolving over centuries into the fresh raw fish preparation known today.', 'topic' => 'food' ),
	array( 'id' => 'food-4', 'text' => 'Vertical farming grows crops in stacked layers inside controlled environments, using LED lights and hydroponics to produce food year-round with minimal water.', 'topic' => 'food' ),

	// Sports
	array( 'id' => 'sport-1', 'text' => 'Marathon running requires months of progressive training that builds aerobic endurance, muscular strength, and mental resilience for the 42-kilometer distance.', 'topic' => 'sports' ),
	array( 'id' => 'sport-2', 'text' => 'Data analytics in professional basketball has revolutionized strategy, with teams using player tracking data to optimize lineups and shot selection.', 'topic' => 'sports' ),
	array( 'id' => 'sport-3', 'text' => 'Rock climbing combines physical strength, flexibility, and problem-solving skills as climbers navigate routes on natural rock faces or indoor walls.', 'topic' => 'sports' ),

	// Environment
	array( 'id' => 'env-1', 'text' => 'Solar panel efficiency has increased dramatically while costs have dropped, making renewable energy competitive with fossil fuels in many regions.', 'topic' => 'environment' ),
	array( 'id' => 'env-2', 'text' => 'Ocean acidification caused by absorbed CO2 threatens coral reefs and marine ecosystems by making it harder for organisms to build calcium carbonate shells.', 'topic' => 'environment' ),
	array( 'id' => 'env-3', 'text' => 'Reforestation projects aim to restore degraded land by planting native tree species, helping to sequester carbon and rebuild biodiversity habitats.', 'topic' => 'environment' ),

	// Education
	array( 'id' => 'edu-1', 'text' => 'Online learning platforms like Coursera and Khan Academy have democratized education by providing free access to courses from top universities worldwide.', 'topic' => 'education' ),
	array( 'id' => 'edu-2', 'text' => 'Project-based learning engages students in real-world problems, developing critical thinking, collaboration, and communication skills beyond textbook knowledge.', 'topic' => 'education' ),
);

echo "PHP Vector Store — Real Embedding Benchmark\n";
echo "=============================================\n";
echo "Model: EmbeddingGemma-300m (via Cloudflare Workers AI)\n";
echo "Dimensions: 768\n";
echo "Dataset: " . count( $dataset ) . " texts across " . count( array_unique( array_column( $dataset, 'topic' ) ) ) . " topics\n\n";

$dir   = sys_get_temp_dir() . '/php-vecstore-real-' . uniqid();
$store = new VectorStore( $dir, 768 );

// ── Generate embeddings via Cloudflare ──────────────────────────────
echo "Generating embeddings...\n";
$embed_start = microtime( true );
$embedded    = 0;

foreach ( $dataset as $item ) {
	$vector = cf_ai_embed( $item['text'] );
	if ( is_wp_error( $vector ) ) {
		echo "  ERROR [{$item['id']}]: " . $vector->get_error_message() . "\n";
		continue;
	}
	$store->set( 'real', $item['id'], $vector, array(
		'topic' => $item['topic'],
		'text'  => substr( $item['text'], 0, 80 ) . '...',
	));
	$embedded++;
	if ( $embedded % 10 === 0 ) {
		echo "  Embedded $embedded/" . count( $dataset ) . "\n";
	}
}

$store->flush();
$embed_ms = round( ( microtime( true ) - $embed_start ) * 1000 );
echo "Embedded $embedded texts in {$embed_ms}ms (" . round( $embed_ms / $embedded ) . "ms/text)\n\n";

// ── Semantic queries ────────────────────────────────────────────────
$queries = array(
	array(
		'text'     => 'How do neural networks learn to recognize images?',
		'expected' => 'tech-1', // ML classifies images
	),
	array(
		'text'     => 'What happens when a star dies?',
		'expected' => 'sci-2', // Black holes
	),
	array(
		'text'     => 'How to make bread at home from scratch',
		'expected' => 'food-1', // Sourdough
	),
	array(
		'text'     => 'Is working from home here to stay?',
		'expected' => 'biz-3', // Remote work
	),
	array(
		'text'     => 'How can we fight drug-resistant bacteria?',
		'expected' => 'med-3', // Antibiotic resistance
	),
	array(
		'text'     => 'AI creating art and pictures',
		'expected' => 'art-1', // Generative AI art
	),
	array(
		'text'     => 'How does gene editing work?',
		'expected' => 'sci-1', // CRISPR
	),
	array(
		'text'     => 'Training for a long distance race',
		'expected' => 'sport-1', // Marathon
	),
	array(
		'text'     => 'Clean energy from the sun',
		'expected' => 'env-1', // Solar panels
	),
	array(
		'text'     => 'Free online courses from universities',
		'expected' => 'edu-1', // Online learning
	),
);

echo "=== Search Accuracy ===\n\n";

// Embed queries
$query_vectors = array();
foreach ( $queries as &$q ) {
	$qv = cf_ai_embed( $q['text'] );
	if ( is_wp_error( $qv ) ) {
		echo "  Query embed error: " . $qv->get_error_message() . "\n";
		continue;
	}
	$q['vector'] = $qv;
	$query_vectors[] = $q;
}
unset( $q );

// Test all search methods
$methods = array(
	'Brute-force 768d' => function ( $store, $q ) {
		return $store->search( 'real', $q['vector'], 3 );
	},
	'Matryoshka 128→384→768' => function ( $store, $q ) {
		return $store->matryoshkaSearch( 'real', $q['vector'], 3, [128, 384, 768] );
	},
);

// Build IVF
$k   = max( 5, (int) ceil( sqrt( $embedded ) ) );
$ivf = new IVFIndex( $store, $k, max( 3, (int) ceil( $k * 0.3 ) ) );
echo "Building IVF index (K=$k)...\n";
$build_info = $ivf->build( 'real', 128 );
echo "Built in {$build_info['build_ms']}ms (avg cluster: {$build_info['avg_cluster']})\n\n";

$methods['IVF (K=' . $k . ')'] = function ( $store, $q ) use ( $ivf ) {
	return $ivf->search( 'real', $q['vector'], 3 );
};
$methods['IVF + Matryoshka'] = function ( $store, $q ) use ( $ivf ) {
	return $ivf->matryoshkaSearch( 'real', $q['vector'], 3, [128, 384, 768] );
};

foreach ( $methods as $method_name => $search_fn ) {
	$hits       = 0;
	$total_ms   = 0;
	$top1_hits  = 0;

	echo "--- $method_name ---\n";

	foreach ( $query_vectors as $q ) {
		$t       = microtime( true );
		$results = $search_fn( $store, $q );
		$ms      = ( microtime( true ) - $t ) * 1000;
		$total_ms += $ms;

		$result_ids = array_map( fn( $r ) => $r['id'], $results );
		$found      = in_array( $q['expected'], $result_ids, true );
		$top1       = ! empty( $results ) && $results[0]['id'] === $q['expected'];

		if ( $found ) $hits++;
		if ( $top1 ) $top1_hits++;

		$status = $top1 ? 'TOP1' : ( $found ? 'TOP3' : 'MISS' );
		$score  = ! empty( $results ) ? $results[0]['score'] : 0;

		printf( "  [%s] Q: %-45s → %s (%.3f)\n",
			$status,
			substr( $q['text'], 0, 45 ),
			$results[0]['id'] ?? '?',
			$score
		);
	}

	$recall_3 = round( $hits / count( $query_vectors ) * 100 );
	$recall_1 = round( $top1_hits / count( $query_vectors ) * 100 );
	$avg_ms   = round( $total_ms / count( $query_vectors ), 2 );

	echo "  Recall@1: {$recall_1}% | Recall@3: {$recall_3}% | Avg: {$avg_ms}ms/query\n\n";
}

// ── Storage stats ───────────────────────────────────────────────────
echo "=== Storage ===\n";
$stats = $store->stats();
echo "  Vectors: {$stats['total_vectors']}\n";
echo "  Size: {$stats['total_bytes']} bytes ({$stats['memory_mb']} MB)\n";
echo "  Per vector: {$stats['bytes_per_vec']} bytes\n";

// Cleanup
$store->drop( 'real' );
$ivf->dropIndex( 'real' );
array_map( 'unlink', glob( "$dir/*" ) );
@rmdir( $dir );

echo "\nDone.\n";
