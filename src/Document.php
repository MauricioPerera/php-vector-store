<?php

namespace PHPVectorStore;

final class Document
{
	/**
	 * @param string               $id       Unique identifier.
	 * @param float[]              $vector   Dense embedding vector.
	 * @param string|null          $text     Raw text for BM25 indexing.
	 * @param array<string, mixed> $metadata Arbitrary key-value payload.
	 */
	public function __construct(
		public readonly string $id,
		public readonly array $vector = [],
		public readonly ?string $text = null,
		public readonly array $metadata = [],
	) {}
}
