<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Stringable;

/**
 * Minkowski
 *
 * The Minkowski distance can be considered as a generalization of both the
 * Euclidean and Manhattan distances. When the lambda parameter is set to 1
 * or 2, the distance is equivalent to Manhattan and Euclidean respectively.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Minkowski implements Distance, Stringable
{
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
     * @param float $lambda
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $lambda = 3.0)
    {
        if ($lambda < 1.0) {
            throw new InvalidArgumentException('Lambda cannot be less'
                . ' than 1.');
        }

        $this->lambda = $lambda;
        $this->inverse = 1.0 / $lambda;
    }

    /**
     * Return the data types that this kernel is compatible with.
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Compute the distance given two vectors.
     *
     * @param list<int|float> $a
     * @param list<int|float> $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0.0;

        foreach ($a as $i => $value) {
            $distance += abs($value - $b[$i]) ** $this->lambda;
        }

        return $distance ** $this->inverse;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Minkowski (lambda: {$this->lambda})";
    }
}
