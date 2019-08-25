<?php

namespace Rubix\ML\Kernels\Distance;

/**
 * Euclidean
 *
 * This is the rotationally invariant straight line (*bee* line) distance
 * between two points.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Euclidean implements Distance
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

        foreach ($a as $i => $value) {
            $distance += ($value - $b[$i]) ** 2;
        }

        return sqrt($distance);
    }
}
