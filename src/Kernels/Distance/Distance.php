<?php

namespace Rubix\ML\Kernels\Distance;

interface Distance
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array;

    /**
     * Compute the distance between two vectors.
     *
     * @param (string|int|float)[] $a
     * @param (string|int|float)[] $b
     * @return float
     */
    public function compute(array $a, array $b) : float;
}
