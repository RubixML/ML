<?php

namespace Rubix\ML\Kernels\Distance;

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
     * Compute the distance between two coordinate vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0;

        foreach ($a as $i => $coordinate) {
            if ($coordinate !== $b[$i]) {
                $distance++;
            }
        }

        $dims = count($a);

        return $dims !== 0 ? (float) ($distance / $dims) : 1.0;
    }
}
