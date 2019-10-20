<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;

use function is_float;
use function is_nan;
use function count;

/**
 * Safe Euclidean
 *
 * An Euclidean distance metric suitable for samples that may contain NaN
 * (not a number) values i.e. missing data. The Safe Euclidean metric approximates
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
class SafeEuclidean implements Distance, NaNSafe
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
        ];
    }

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

            switch (true) {
                case is_float($valueA) and is_nan($valueA):
                    $nn++;

                    break 1;

                case is_float($valueB) and is_nan($valueB):
                    $nn++;

                    break 1;

                default:
                    $distance += ($valueA - $valueB) ** 2;
            }
        }

        if ($nn === $n) {
            return NAN;
        }

        $scale = $n / ($n - $nn);

        return sqrt($scale * $distance);
    }
}
