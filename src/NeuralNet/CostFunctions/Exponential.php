<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

/**
 * Exponential
 *
 * This cost function calculates the exponential of a prediction's squared error
 * thus applying a large penalty to wrong predictions. The resulting gradient of
 * the Exponential loss tends to be steeper than most other cost functions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Exponential implements CostFunction
{
    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return array
     */
    public function range() : array
    {
        return [0.0, INF];
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
        return exp(($expected - $activation) ** 2);
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
        return ($expected - $activation) * $computed;
    }
}
