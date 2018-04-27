<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

interface ActivationFunction
{
    /**
     * Compute the activation of the neuron.
     *
     * @param  float  $value
     * @return float
     */
    public function compute(float $value) : float;

    /**
     * Compute the derivative.
     *
     * @param  float  $value
     * @param  float  $computed
     * @return float
     */
    public function differentiate(float $value, float $computed) : float;
}
