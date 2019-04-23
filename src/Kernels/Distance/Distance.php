<?php

namespace Rubix\ML\Kernels\Distance;

interface Distance
{
    /**
     * Compute the distance between two vectors.
     *
     * @param array $a
     * @param array $b
     * @return float
     */
    public function compute(array $a, array $b) : float;
}
