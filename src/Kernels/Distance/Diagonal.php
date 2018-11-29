<?php

namespace Rubix\ML\Kernels\Distance;

/**
 * Diagonal
 *
 * The Diagonal (sometimes called Chebyshev) distance is a measure that
 * constrains movement to horizontal, vertical, and diagonal from a point. An
 * example that uses Diagonal movement is a chess board.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Diagonal implements Distance
{
    /**
     * Compute the distance between two vectors.
     *
     * @param  array  $a
     * @param  array  $b
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
