<?php

namespace PHPVectorStore\Exception;

class DimensionMismatchException extends VectorStoreException
{
	public static function forVectors( int $expected, int $actual ): self {
		return new self( sprintf(
			'Dimension mismatch: expected %d, got %d.',
			$expected,
			$actual,
		) );
	}
}
