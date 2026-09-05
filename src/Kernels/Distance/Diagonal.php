<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;

/**
 * Diagonal
 *
 * The Diagonal (a.k.a. *Chebyshev*) distance is a measure that constrains
 * movement to horizontal, vertical, and diagonal movement from a point. An
 * example of a game that uses diagonal movement is a chess board.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Diagonal implements Distance, Subadditive, Monotonic
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @internal
     *
     * @return list<DataType>
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
     * @param list<float> $a
     * @param list<float> $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0.0;

        foreach ($a as $i => $value) {
            $distance = max($distance, abs($value - $b[$i]));
        }

        return $distance;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Diagonal';
    }
}
