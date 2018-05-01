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
     * @param  \Rubix\Engine\NeuralNet\Synapse  $synapse
     * @param  int  $n
     * @return array
     */
    public function initialize(int $n) : array
    {
        return [-3, 3];
    }
}
