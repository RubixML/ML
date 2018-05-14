<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

class Identity implements ActivationFunction
{
    /**
     * Compute the indentity of the input. i.e. the input value is the output.
     *
     * @param  float  $value
     * @return float
     */
    public function compute(float $value) : float
    {
        return $value;
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
        return 1.0;
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

        return random_int(-3 * $scale, 3 * $scale) / $scale;
    }
}
