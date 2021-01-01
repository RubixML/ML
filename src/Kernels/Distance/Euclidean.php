<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;

/**
 * Euclidean
 *
 * Standard straight line (*bee* line) distance between two points. The Euclidean
 * distance has the nice property of being invariant under any rotation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Euclidean implements Distance
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Compute the distance between two vectors.
     *
     * @internal
     *
     * @param list<int|float> $a
     * @param list<int|float> $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0.0;

        foreach ($a as $i => $value) {
            $distance += ($value - $b[$i]) ** 2;
        }

        return sqrt($distance);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Euclidean';
    }
}
