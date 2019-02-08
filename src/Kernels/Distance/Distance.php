<?php

namespace Rubix\ML\Kernels\Distance;

interface Distance
{
    const EPSILON = 1e-8;

    /**
     * Return a list of data types distance is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array;
    
    /**
     * Compute the distance between two vectors.
     *
     * @param array $a
     * @param array $b
     * @return float
     */
    public function compute(array $a, array $b) : float;
}
