<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

interface DistanceFunction
{
    /**
     * Compute the distance between given two coordinate vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @return float
     */
    public function compute(array $a, array $b) : float;
}
