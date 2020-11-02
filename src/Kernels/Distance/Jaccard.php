<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;

use const Rubix\ML\EPSILON;

/**
 * Jaccard
 *
 * The *generalized* Jaccard distance is a measure of distance with a range from 0 to
 * 1 and can be thought of as the size of the intersection divided by the size of the
 * union of two points if they were consisted only of binary random variables.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Jaccard implements Distance
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
        $distance = $min = $max = 0.0;

        foreach ($a as $i => $valueA) {
            $valueB = $b[$i];

            $min += min($valueA, $valueB);
            $max += max($valueA, $valueB);
        }

        return 1.0 - ($min / ($max ?: EPSILON));
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Jaccard';
    }
}
