<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\Datasets\DataFrame;

/**
 * Hamming
 *
 * The Hamming distance is defined as the sum of all coordinates that are not
 * exactly the same. Therefore, two coordinate vectors a and b would have a
 * Hamming distance of 2 if only one of the three coordinates were equal between
 * the vectors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Hamming implements Distance
{
    /**
     * Return a list of data types distance is compatible with.
     * 
     * @var int[]
     */
    public function compatibility() : array
    {
        return [
            DataFrame::CATEGORICAL,
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
        $distance = 0;

        foreach ($a as $i => $value) {
            if ($value !== $b[$i]) {
                $distance++;
            }
        }

        return $distance / (count($a) ?: self::EPSILON);
    }
}
