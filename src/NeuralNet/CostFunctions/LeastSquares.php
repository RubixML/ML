<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

/**
 * Least Squares
 *
 * Least Squares or *quadratic* loss is a function that measures the squared
 * error between the target output and the actual output of a network.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeastSquares implements CostFunction
{
    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return array
     */
    public function range() : array
    {
        return [0, INF];
    }

    /**
     * Compute the cost.
     *
     * @param  float  $expected
     * @param  float  $activation
     * @return float
     */
    public function compute(float $expected, float $activation) : float
    {
        return 0.5 * ($activation - $expected) ** 2;
    }

    /**
     * Calculate the derivative of the cost function.
     *
     * @param  float  $expected
     * @param  float  $activation
     * @param  float  $computed
     * @return float
     */
    public function differentiate(float $expected, float $activation, float $computed) : float
    {
        return $activation - $expected;
    }
}
