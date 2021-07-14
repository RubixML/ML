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
class Diagonal implements Distance
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
        $deltas = [];

        foreach ($a as $i => $value) {
            $deltas[] = abs($value - $b[$i]);
        }

        return max($deltas) ?: 0.0;
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
