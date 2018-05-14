<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

class Gaussian implements ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @param  float  $value
     * @return float
     */
    public function compute(float $value) : float
    {
        return exp(-($value ** 2));
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
        return -2 * $value * $computed;
    }
    
    /**
     * Generate an initial synapse weight range based on n, the number of inputs
     * to a particular neuron.
     *
     * @param  int  $inDegree
     * @return float
     */
    public function initialize(int $inDegree) : float
    {
        $scale = pow(10, 10);

        return random_int(-4 * $scale, 4 * $scale) / $scale;
    }
}
