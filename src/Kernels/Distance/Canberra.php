<?php

namespace Rubix\ML\Kernels\Distance;

use const Rubix\ML\EPSILON;

/**
 * Canberra
 *
 * A weighted version of Manhattan distance which computes the L1 distance
 * between two coordinates in a vector space.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Canberra implements Distance
{
    /**
     * Compute the distance between two vectors.
     *
     * @param array $a
     * @param array $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0.;

        foreach ($a as $i => $valueA) {
            $valueB = $b[$i];

            $distance += abs($valueA - $valueB)
                / ((abs($valueA) + abs($valueB)) ?: EPSILON);
        }

        return $distance;
    }
}
