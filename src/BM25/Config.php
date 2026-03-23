<?php

namespace PHPVectorStore\BM25;

final class Config
{
	public function __construct(
		public readonly float $k1 = 1.5,
		public readonly float $b = 0.75,
	) {
		if ( $k1 < 0 ) {
			throw new \InvalidArgumentException( 'k1 must be >= 0.' );
		}
		if ( $b < 0.0 || $b > 1.0 ) {
			throw new \InvalidArgumentException( 'b must be between 0 and 1.' );
		}
	}
}
