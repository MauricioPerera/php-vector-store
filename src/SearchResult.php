<?php

namespace PHPVectorStore;

final class SearchResult
{
	/**
	 * @param string      $id         Document identifier.
	 * @param float       $score      Relevance score (higher = better).
	 * @param int         $rank       1-based rank in result list.
	 * @param array       $metadata   Document metadata.
	 * @param string|null $collection Collection name (multi-collection search).
	 * @param int|null    $cluster    IVF cluster ID.
	 * @param array       $stages     Matryoshka per-stage scores.
	 */
	public function __construct(
		public readonly string $id,
		public readonly float $score,
		public readonly int $rank,
		public readonly array $metadata = [],
		public readonly ?string $collection = null,
		public readonly ?int $cluster = null,
		public readonly array $stages = [],
	) {}
}
