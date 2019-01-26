<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\Datasets\DataFrame;
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
     * This parameter controls the *roundedness* of the metric. There are
     * special cases when lambda = 1 then it is equivalent to manhattan
     * distance, when lambda = 2 it is equivalent to euclidean distance.
     * 
     * @var float
     */
    protected $lambda;

    /**
     * The inverse of the lambda parameter.
     * 
     * @var float
     */
    protected $inverse;

    /**
     * @param  float  $lambda
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $lambda = 3.)
    {
        if ($lambda < 1.) {
            throw new InvalidArgumentException('Lambda cannot be less'
                . ' than 1.');
        }

        $this->lambda = $lambda;
        $this->inverse = 1. / $lambda;
    }

    /**
     * Compute the distance given two vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0.;

        foreach ($a as $i => $value) {
            $distance += abs($value - $b[$i]) ** $this->lambda;
        }

        return $distance ** $this->inverse;
    }
}
