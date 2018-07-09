<?php

namespace Rubix\ML\Kernels\Distance;

class Hamming implements Distance
{
    /**
     * Compute the distance between two coordinate vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0;

        foreach ($a as $i => $coordinate) {
            if ($coordinate !== $b[$i]) {
                $distance++;
            }
        }

        return (float) $distance;
    }
}
