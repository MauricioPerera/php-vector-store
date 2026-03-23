<?php

namespace PHPVectorStore\BM25;

interface TokenizerInterface
{
	/** @return string[] */
	public function tokenize( string $text ): array;
}
