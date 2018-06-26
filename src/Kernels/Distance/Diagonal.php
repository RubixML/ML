<?php

namespace Rubix\ML\Kernels\Distance;

class Diagonal implements Distance
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
        $distances = [];

        foreach ($a as $i => $coordinate) {
            $distances[] = abs($coordinate - $b[$i]);
        }

        return max($distances);
    }
}
