<?php

namespace Rubix\Engine\NeuralNetwork\ActivationFunctions;

class Heaviside implements ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @param  float  $value
     * @return float
     */
    public function compute(float $value) : float
    {
        return $value >= 0.0 ? 1.0 : 0.0;
    }

    /**
     * Compute the derivative.
     *
     * @param  float  $value
     * @param  float  $computed
     * @return float
     */
    public function differentiate(float $value, float $computed) : float
    {
        return $value === 0.0 ? 1.0 : 0.0;
    }
}
