<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

class Softsign implements ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @param  float  $value
     * @return float
     */
    public function compute(float $value) : float
    {
        return $value / (1 + abs($value));
    }

    /**
     * Calculate the partial derivative with respect to the computed output.
     *
     * @param  float  $value
     * @param  float  $computed
     * @return float
     */
    public function differentiate(float $value, float $computed) : float
    {
        return 1 / (1 + abs($computed)) ** 2;
    }
}
