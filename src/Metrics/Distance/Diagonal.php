<?php

namespace Rubix\Engine\Metrics\Distance;

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
        return (float) max(array_map(function ($ca, $cb) {
            return abs($ca - $cb);
        }, $a, $b));
    }
}
