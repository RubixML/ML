<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;

use function count;
use function is_float;
use function is_nan;

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
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
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
     * @internal
     *
     * @param list<int|float> $a
     * @param list<int|float> $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0.0;
        $nn = 0;

        foreach ($a as $i => $valueA) {
            $valueB = $b[$i];

            if (is_float($valueA) and is_nan($valueA)) {
                ++$nn;

                continue;
            }

            if (is_float($valueB) and is_nan($valueB)) {
                ++$nn;

                continue;
            }

            $distance += ($valueA - $valueB) ** 2;
        }

        $n = count($a);

        if ($nn === $n) {
            return NAN;
        }

        return sqrt($n / ($n - $nn) * $distance);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Safe Euclidean';
    }
}
