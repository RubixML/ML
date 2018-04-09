<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

use InvalidArgumentException;

class Euclidean implements DistanceFunction
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

        $distance = 0.0;

        foreach ($a as $i => $coordinate) {
            $distance += ($coordinate - $b[$i]) ** 2;
        }

        return sqrt($distance);
    }
}
