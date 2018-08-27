<?php

namespace Rubix\ML\Kernels\Distance;

use InvalidArgumentException;

/**
 * Minkowski
 *
 * The Minkowski distance is a metric in a normed vector space which can be
 * considered as a generalization of both the Euclidean and Manhattan distances.
 * When the lambda parameter is set to 1 or 2, the distance is equivalent to
 * Manhattan and Euclidean respectively.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Minkowski implements Distance
{
    /**
     * @var float
     */
    protected $lambda;

    /**
     * @var float
     */
    protected $exponent;

    /**
     * @param  float  $lambda
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $lambda = 3.)
    {
        if ($lambda <= 0.) {
            throw new InvalidArgumentException('Lambda must be greater than'
                . ' 0.');
        }

        $this->lambda = $lambda;
        $this->exponent = 1. / $this->lambda;
    }

    /**
     * Compute the distance given two coordinate vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0.;

        foreach ($a as $i => $coordinate) {
            $distance += abs($coordinate - $b[$i]) ** $this->lambda;
        }

        return $distance ** $this->exponent;
    }
}
