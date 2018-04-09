<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

use InvalidArgumentException;

class Diagonal implements DistanceFunction
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

        return (float) max(array_map(function ($ca, $cb) {
            return abs($ca - $cb);
        }, $a, $b));
    }
}
