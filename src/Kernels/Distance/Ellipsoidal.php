<?php

namespace Rubix\ML\Kernels\Distance;

use MathPHP\LinearAlgebra\Vector;
use InvalidArgumentException;

/**
 * Ellipsoidal
 *
 * The Ellipsoidal distance measures the distance between two points on a
 * 3-dimensional ellipsoid.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Ellipsoidal implements Distance
{
    /**
     * Compute the distance between two coordinate vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @throws \InvalidArgumentException
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        if (count($a) !== 3 and count($b) !== 3) {
            throw new InvalidArgumentException('This distance kernel only works'
                . ' in 3 dimensions.');
        }

        $a = new Vector(array_values($a));
        $b = new Vector(array_values($b));

        return atan($a->crossProduct($b)->length() / $a->dotProduct($b));
    }
}
