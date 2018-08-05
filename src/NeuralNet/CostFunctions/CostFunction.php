<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

interface CostFunction
{
    const EPSILON = 1e-8;

    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return array
     */
    public function range() : array;

    /**
     * Compute the cost.
     *
     * @param  float  $expected
     * @param  float  $activation
     * @return float
     */
    public function compute(float $expected, float $activation) : float;

    /**
     * Calculate the derivative of the cost function.
     *
     * @param  float  $expected
     * @param  float  $activation
     * @param  float  $computed
     * @return float
     */
    public function differentiate(float $expected, float $activation, float $computed) : float;
}
