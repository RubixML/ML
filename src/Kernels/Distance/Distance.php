<?php

namespace Rubix\ML\Kernels\Distance;

/**
 * Distance
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Distance
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array;

    /**
     * Compute the distance between two vectors.
     *
     * @internal
     *
     * @param list<string|int|float> $a
     * @param list<string|int|float> $b
     * @return float
     */
    public function compute(array $a, array $b) : float;

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
