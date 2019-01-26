<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\Datasets\DataFrame;

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
        $distance = 0.;

        foreach ($a as $i => $value) {
            $distance += abs($value - $b[$i]);
        }

        return $distance;
    }
}
