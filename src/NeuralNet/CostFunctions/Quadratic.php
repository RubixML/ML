<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

class Quadratic implements CostFunction
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
        return 0.5 * ($expected - $activation) ** 2;
    }

    /**
     * Calculate the derivative of the cost function.
     *
     * @param  float  $expected
     * @param  float  $activation
     * @return float
     */
    public function differentiate(float $expected, float $activation) : float
    {
        return $expected - $activation;
    }
}
