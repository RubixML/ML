<?php

namespace Rubix\Engine\Metrics\Distance;

class Manhattan implements Distance
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
        $distance = 0.0;

        foreach ($a as $index => $coordinate) {
            $distance += abs($coordinate - $b[$index]);
        }

        return $distance;
    }
}
