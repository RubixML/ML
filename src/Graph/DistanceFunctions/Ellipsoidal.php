<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

use MathPHP\LinearAlgebra\Vector;
use InvalidArgumentException;

class Ellipsoidal implements DistanceFunction
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
        if (count($a) !== 3 && count($b) !== 3) {
            throw new InvalidArgumentException('This distance function only works in 3 dimensions.');
        }

        $a = new Vector(array_values($a));
        $b = new Vector(array_values($b));

        return atan($a->crossProduct($b)->length() / $a->dotProduct($b));
    }
}
