<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

class SoftPlus implements ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @param  float  $value
     * @return float
     */
    public function compute(float $value) : float
    {
        return log(1 + exp($value));
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
        return 1 / (1 + exp(-$computed));
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
        $r = pow(6 / $inDegree, 1 / self::ROOT_2);

        $scale = pow(10, 10);

        return random_int(-$r * $scale, $r * $scale) / $scale;
    }
}
