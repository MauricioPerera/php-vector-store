<?php
/**
 * Neuron AI adapter — implements VectorStoreInterface for the Neuron AI framework.
 *
 * Zero-dependency local vector store for Neuron AI RAG agents.
 * Replaces Pinecone, Qdrant, Chroma, etc. with pure PHP binary storage.
 *
 * Usage:
 *   class MyRAG extends RAG {
 *       protected function vectorStore(): VectorStoreInterface {
 *           return new NeuronVectorStore( __DIR__ . '/vectors', 384 );
 *       }
 *   }
 *
 * @package PHPVectorStore
 */

namespace PHPVectorStore\Integration;

use NeuronAI\RAG\VectorStore\VectorStoreInterface;
use NeuronAI\RAG\Document;
use PHPVectorStore\StoreInterface;
use PHPVectorStore\VectorStore;
use PHPVectorStore\QuantizedStore;

class NeuronVectorStore implements VectorStoreInterface
{
	private StoreInterface $store;
	private string $collection;
	private int $topK;
	private bool $matryoshka;
	private array $stages;

	/**
	 * @param string $directory   Storage directory.
	 * @param int    $dimensions  Vector dimensions (768, 384, etc.).
	 * @param string $collection  Collection name (default 'documents').
	 * @param int    $topK        Number of results for similarity search.
	 * @param bool   $quantized   Use Int8 quantization (4x smaller, same accuracy).
	 * @param bool   $matryoshka  Use Matryoshka multi-stage search.
	 * @param int[]  $stages      Matryoshka dimension stages.
	 */
	public function __construct(
		string $directory,
		int    $dimensions = 384,
		string $collection = 'documents',
		int    $topK = 4,
		bool   $quantized = true,
		bool   $matryoshka = true,
		array  $stages = array(),
	) {
		$this->store      = $quantized
			? new QuantizedStore( $directory, $dimensions )
			: new VectorStore( $directory, $dimensions );
		$this->collection = $collection;
		$this->topK       = $topK;
		$this->matryoshka = $matryoshka;
		$this->stages     = $stages ?: self::defaultStages( $dimensions );
	}

	/**
	 * Add a single document.
	 */
	public function addDocument( Document $document ): VectorStoreInterface {
		$this->store->set(
			$this->collection,
			(string) $document->id,
			$document->embedding,
			array(
				'content'    => $document->content,
				'sourceType' => $document->sourceType,
				'sourceName' => $document->sourceName,
				'metadata'   => $document->metadata,
			)
		);
		$this->store->flush();
		return $this;
	}

	/**
	 * Add multiple documents in batch.
	 */
	public function addDocuments( array $documents ): VectorStoreInterface {
		foreach ( $documents as $doc ) {
			$this->store->set(
				$this->collection,
				(string) $doc->id,
				$doc->embedding,
				array(
					'content'    => $doc->content,
					'sourceType' => $doc->sourceType,
					'sourceName' => $doc->sourceName,
					'metadata'   => $doc->metadata,
				)
			);
		}
		$this->store->flush();
		return $this;
	}

	/**
	 * Delete documents by source.
	 */
	public function deleteBy( string $sourceType, ?string $sourceName = null ): VectorStoreInterface {
		$ids = $this->store->ids( $this->collection );

		foreach ( $ids as $id ) {
			$record = $this->store->get( $this->collection, $id );
			if ( ! $record ) continue;

			$meta = $record['metadata'] ?? array();
			if ( ( $meta['sourceType'] ?? '' ) !== $sourceType ) continue;

			if ( null !== $sourceName && ( $meta['sourceName'] ?? '' ) !== $sourceName ) continue;

			$this->store->remove( $this->collection, $id );
		}

		$this->store->flush();
		return $this;
	}

	/**
	 * Similarity search — returns Documents ordered by score descending.
	 *
	 * @param float[] $embedding Query embedding vector.
	 * @return Document[]
	 */
	public function similaritySearch( array $embedding ): iterable {
		$results = $this->matryoshka
			? $this->store->matryoshkaSearch( $this->collection, $embedding, $this->topK, $this->stages )
			: $this->store->search( $this->collection, $embedding, $this->topK );

		$documents = array();

		foreach ( $results as $r ) {
			$meta = $r['metadata'] ?? array();

			$doc             = new Document( $meta['content'] ?? '' );
			$doc->id         = $r['id'];
			$doc->score      = $r['score'];
			$doc->sourceType = $meta['sourceType'] ?? 'manual';
			$doc->sourceName = $meta['sourceName'] ?? 'manual';
			$doc->metadata   = $meta['metadata'] ?? array();
			$doc->embedding  = array(); // Don't return full embedding to save memory

			$documents[] = $doc;
		}

		return $documents;
	}

	/**
	 * Get the underlying store instance.
	 */
	public function getStore(): StoreInterface {
		return $this->store;
	}

	/**
	 * Default Matryoshka stages based on dimensions.
	 */
	private static function defaultStages( int $dim ): array {
		if ( $dim <= 128 ) return array( $dim );
		if ( $dim <= 256 ) return array( 128, $dim );
		if ( $dim <= 384 ) return array( 128, 256, $dim );
		return array( 128, 384, $dim );
	}
}
