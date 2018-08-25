<?php

namespace Rubix\ML\Kernels\Distance;

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
     * Compute the distance between two coordinate vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0.;

        foreach ($a as $i => $coordinate) {
            $distance += abs($coordinate - $b[$i])
                / (abs($coordinate) + abs($b[$i]));
        }

        return $distance;
    }
}
