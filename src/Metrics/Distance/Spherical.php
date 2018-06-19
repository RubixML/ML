<?php

namespace Rubix\ML\Metrics\Distance;

use InvalidArgumentException;

class Spherical implements Distance
{
    /**
     * The radius of the sphere.
     *
     * @var float
     */
    protected $radius;

    /**
     * @param  float  $radius
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $radius = 6371.0)
    {
        if ($radius <= 0) {
            throw new InvalidArgumentException('Radius must be greater than'
                . ' 0.');
        }

        $this->radius = $radius;
    }

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
        if (count($a) !== 2 and count($b) !== 2) {
            throw new InvalidArgumentException('This distance function only'
                . ' works in 2 dimensions.');
        }

        $a = array_values($a);
        $b = array_values($b);

        $d = [deg2rad($b[0] - $a[0]), deg2rad($b[1] - $a[1])];

        $c = sin($d[0] / 2) * sin($d[0] / 2)
            + cos(deg2rad($a[0])) * cos(deg2rad($b[0]))
            * sin($d[1] / 2) * sin($d[1] / 2);

        return $this->radius * 2 * asin(sqrt($c));
    }
}
