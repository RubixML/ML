<?php

namespace Rubix\ML\Kernels\Distance;

/**
 * Manhattan
 *
 * A distance metric that constrains movement to horizontal and vertical,
 * similar to navigating the city blocks of Manhattan. An example that used this
 * type of movement is a checkers board.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Manhattan implements Distance
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
        $distance = 0.0;

        foreach ($a as $i => $coordinate) {
            $distance += abs($coordinate - $b[$i]);
        }

        return $distance;
    }
}
