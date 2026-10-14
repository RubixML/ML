<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;

/**
 * Squared Euclidean
 *
 * The squared Euclidean distance between two points, computed as the sum of
 * the squared differences of their features without the final square root
 * applied by standard [Euclidean](euclidean.md). This distance is not a
 * metric — it does not satisfy the triangle inequality — but it is symmetric
 * and coordinatewise monotone.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SquaredEuclidean implements Distance, Monotonic, Symmetric
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @internal
     *
     * @return list<DataType>
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
     * @param list<float> $a
     * @param list<float> $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0.0;

        foreach ($a as $i => $value) {
            $delta = $value - $b[$i];

            $distance += $delta * $delta;
        }

        return $distance;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Squared Euclidean';
    }
}
