<?php

namespace PHPVectorStore;

enum Distance
{
	case Cosine;
	case Euclidean;
	case DotProduct;
	case Manhattan;
}
