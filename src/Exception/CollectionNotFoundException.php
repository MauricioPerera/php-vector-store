<?php

namespace PHPVectorStore\Exception;

class CollectionNotFoundException extends VectorStoreException
{
	public static function forCollection( string $name ): self {
		return new self( sprintf( 'Collection "%s" not found.', $name ) );
	}
}
