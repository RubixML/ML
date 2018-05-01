<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

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
     * Calculate the partial derivative with respect to the computed output.
     *
     * @param  float  $value
     * @param  float  $computed
     * @return float
     */
    public function differentiate(float $value, float $computed) : float
    {
        return 1 - ($computed ** 2);
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
        $r = pow(6 / $n, 1 / 4);

        return [-$r, $r];
    }
}
