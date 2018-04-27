<?php

namespace Rubix\Engine\Metrics\DistanceFunctions;

use InvalidArgumentException;

class Minkowski implements DistanceFunction
{
    /**
     * @var float
     */
    protected $lambda;

    /**
     * @param  float  $lambda
     * @return void
     */
    public function __construct(float $lambda = 3.0)
    {
        $this->lambda = $lambda;
    }

    /**
     * Compute the distance given two coordinate vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @throws \InvalidArgumentException
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        if (count($a) !== count($b)) {
            throw new InvalidArgumentException('The size of each coordinate vector must be equal.');
        }

        $distance = 0.0;

        foreach ($a as $i => $coordinate) {
            $distance += pow(abs($coordinate - $b[$i]), $this->lambda);
        }

        return (float) pow($distance, 1.0 / $this->lambda);
    }
}
