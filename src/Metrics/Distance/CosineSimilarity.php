<?php

namespace Rubix\ML\Metrics\Distance;

use MathPHP\LinearAlgebra\Vector;

class CosineSimilarity implements Distance
{
    /**
     * Compute the distance between two coordinates.
     *
     * @param  array  $a
     * @param  array  $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $a = new Vector($a);
        $b = new Vector($b);

        return 1 - $a->dotProduct($b) / ($a->length() * $b->length());
    }
}
