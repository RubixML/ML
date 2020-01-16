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
     * @return \Rubix\ML\DataType[]
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
     * @param (int|float)[] $a
     * @param (int|float)[] $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $deltas = [];

        foreach ($a as $i => $value) {
            $deltas[] = abs($value - $b[$i]);
        }

        return max($deltas);
    }
}
