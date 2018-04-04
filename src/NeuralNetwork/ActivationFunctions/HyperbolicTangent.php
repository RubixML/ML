<?php

namespace Rubix\Engine\NeuralNetwork\ActivationFunctions;

class HyperbolicTangent implements ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @param  float  $value
     * @return float
     */
    public function compute(float $value) : float
    {
        return tanh($value);
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
        return 1 - ($computed ** 2);
    }
}
