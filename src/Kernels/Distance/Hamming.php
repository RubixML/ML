<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;

/**
 * Hamming
 *
 * A categorical distance function that measures distance as the number of
 * substitutions necessary to convert one sample to the other.
 *
 * References:
 * [1] R. W. Hamming. (1950). Error detecting and error correcting codes.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Hamming implements Distance
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
            DataType::categorical(),
        ];
    }

    /**
     * Compute the distance between two vectors.
     *
     * @internal
     *
     * @param list<string> $a
     * @param list<string> $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0;

        foreach ($a as $i => $value) {
            if ($value !== $b[$i]) {
                ++$distance;
            }
        }

        return (float) $distance;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Hamming';
    }
}
