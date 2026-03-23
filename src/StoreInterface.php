<?php

namespace PHPVectorStore;

interface StoreInterface
{
	public function dimensions(): int;

	public function directory(): string;

	public function set( string $collection, string $id, array $vector, array $metadata = array() ): void;

	public function remove( string $collection, string $id ): bool;

	/** @return array{id: string, vector: float[], metadata: array}|null */
	public function get( string $collection, string $id ): ?array;

	public function has( string $collection, string $id ): bool;

	public function count( string $collection ): int;

	/** @return string[] */
	public function ids( string $collection ): array;

	/** @return string[] */
	public function collections(): array;

	public function search( string $collection, array $query, int $limit = 5, int $dimSlice = 0, ?Distance $distance = null ): array;

	public function matryoshkaSearch( string $collection, array $query, int $limit = 5, array $stages = array( 128, 384, 768 ), int $candidateMultiplier = 3, ?Distance $distance = null ): array;

	public function searchAcross( array $collections, array $query, int $limit = 5, int $dimSlice = 0, ?Distance $distance = null ): array;

	public function flush(): void;

	public function drop( string $collection ): void;

	public function stats(): array;

	public function import( string $collection, array $records ): int;

	public function export( string $collection ): array;
}
