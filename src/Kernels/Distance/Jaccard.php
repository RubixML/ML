<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\Datasets\DataFrame;

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
     * Return a list of data types distance is compatible with.
     * 
     * @var int[]
     */
    public function compatibility() : array
    {
        return [
            DataFrame::CONTINUOUS,
        ];
    }

    /**
     * Compute the distance between two vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = $mins = $maxs = 0.;

        foreach ($a as $i => $valueA) {
            $valueB = $b[$i];

            $mins += min($valueA, $valueB);
            $maxs += max($valueA, $valueB);
        }

        return 1. - ($mins / ($maxs ?: self::EPSILON));
    }
}
