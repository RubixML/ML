<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

use MathPHP\LinearAlgebra\Vector;
use InvalidArgumentException;

class CosineSimilarity implements DistanceFunction
{
    /**
     * Compute the distance between two coordinate vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @throws \InvalidArgumentException
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        if (count($a) !== count($b)) {
            throw new InvalidArgumentException('The size of each coordinate vector must be equal.');
        }

        $a = new Vector(array_values($a));
        $b = new Vector(array_values($b));

        return 1 - $a->dotProduct($b) / ($a->length() * $b->length());
    }
}
