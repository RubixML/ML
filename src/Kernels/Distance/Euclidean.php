<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\Other\Helpers\DataType;
use function sqrt;

/**
 * Euclidean
 *
 * This is the ordinary straight line (bee line) distance between two points in
 * Euclidean space.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Euclidean implements Distance
{
    /**
     * Return a list of data types distance is compatible with.
     *
     * @var int[]
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
     * @param  array  $a
     * @param  array  $b
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
