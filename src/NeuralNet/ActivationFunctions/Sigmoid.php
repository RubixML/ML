<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

use Rubix\Engine\NeuralNet\Synapse;

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
     * Calculate the partial derivative with respect to the computed output.
     *
     * @param  float  $value
     * @param  float  $computed
     * @return float
     */
    public function differentiate(float $value, float $computed) : float
    {
        return $computed * (1 - $computed);
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
        $r = sqrt(6 / $inDegree);

        $scale = pow(10, 10);

        return random_int(-$r * $scale, $r * $scale) / $scale;
    }
}
