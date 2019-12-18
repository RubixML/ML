<?php

namespace Rubix\ML\Kernels\Distance;

interface Distance
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array;

    /**
     * Compute the distance between two vectors.
     *
     * @param mixed[] $a
     * @param mixed[] $b
     * @return float
     */
    public function compute(array $a, array $b) : float;
}
