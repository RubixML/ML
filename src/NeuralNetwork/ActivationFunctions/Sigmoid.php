<?php

namespace Rubix\Engine\NeuralNetwork\ActivationFunctions;

class Sigmoid implements ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @param  float  $value
     * @return float
     */
    public function compute(float $value) : float
    {
        return 1 / (1 + exp(-$value));
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
        return $computed * (1 - $computed);
    }
}
