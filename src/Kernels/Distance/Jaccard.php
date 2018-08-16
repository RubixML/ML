<?php

namespace Rubix\ML\Kernels\Distance;

/**
 * Jaccard
 *
 * The generalized Jaccard distance is a measure of similarity that one
 * sample has to another with a range from 0 to 1. The higher the percentage,
 * the more dissimilar they are.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Jaccard implements Distance
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
        $distance = $mins = $maxs = 0.0;

        foreach ($a as $i => $coordinate) {
            $mins += min($coordinate, $b[$i]);
            $maxs += max($coordinate, $b[$i]);
        }

        return 1.0 - ($mins / $maxs);
    }
}
