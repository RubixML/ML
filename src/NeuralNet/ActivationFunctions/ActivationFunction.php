<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

use Rubix\Engine\NeuralNet\Synapse;

interface ActivationFunction
{
    const ROOT_2 = 1.41421356237;

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


    /**
     * Generate an initial synapse weight range.
     *
     * @param  int  $inDegree
     * @return float
     */
    public function initialize(int $inDegree) : float;
}
