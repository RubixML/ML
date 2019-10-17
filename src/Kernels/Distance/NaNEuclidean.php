<?php

namespace Rubix\ML\Kernels\Distance;

use function is_nan;

/**
 * NaN Euclidean
 *
 * An Euclidean distance metric suitable for samples that may contain NaN
 * (not a number) values i.e. missing data. The NaN Euclidean metric approximates
 * the Euclidean distance function by dropping NaN values and scaling the distance
 * according to the proportion of non-NaNs (in either a or b or both) to compensate.
 *
 * References:
 * [1] J. K. Dixon. (1978). Pattern Recognition with Partly Missing Data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NaNEuclidean implements Distance
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
        $nn = 0;

        $n = count($a);

        foreach ($a as $i => $valueA) {
            $valueB = $b[$i];

            if (is_nan($valueA) or is_nan($valueB)) {
                $nn++;

                continue 1;
            }

            $distance += ($valueA - $valueB) ** 2;
        }

        $scale = $n / ($n - $nn);

        return sqrt($scale * $distance);
    }
}
