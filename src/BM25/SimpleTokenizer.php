<?php

namespace PHPVectorStore\BM25;

/**
 * Lightweight, language-agnostic tokenizer.
 *
 * Pipeline: mb_strtolower → split Unicode → stop words → min length.
 */
final class SimpleTokenizer implements TokenizerInterface
{
	/** @var array<string, true> */
	private readonly array $stopWords;

	public function __construct(
		array $stopWords = self::DEFAULT_STOP_WORDS,
		private readonly int $minTokenLength = 2,
	) {
		$this->stopWords = array_fill_keys(
			array_map( 'mb_strtolower', $stopWords ),
			true,
		);
	}

	public function tokenize( string $text ): array {
		$text   = mb_strtolower( $text, 'UTF-8' );
		$tokens = preg_split( '/[^\p{L}\p{N}]+/u', $text, -1, PREG_SPLIT_NO_EMPTY ) ?: [];

		$result = [];
		foreach ( $tokens as $token ) {
			if (
				mb_strlen( $token, 'UTF-8' ) >= $this->minTokenLength
				&& ! isset( $this->stopWords[ $token ] )
			) {
				$result[] = $token;
			}
		}
		return $result;
	}

	public const DEFAULT_STOP_WORDS = [
		'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
		'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
		'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
		'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
		'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
		'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
		'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
		'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
		'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
		'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
		'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
		'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
		'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
		'don', 'should', 'now',
	];
}
