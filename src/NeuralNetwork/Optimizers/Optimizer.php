<?php

namespace Rubix\Engine\NeuralNetwork\Optimizers;

use Rubix\Engine\NeuralNetwork\Synapse;

interface Optimizer
{
    const EPSILON = 1e-8;

    /**
     * Calculate the amount of a step of gradient descent.
     *
     * @param  \Rubix\Engine\NeuralNetwork\Synapse  $synapse
     * @param  float  $gradient
     * @return float
     */
    public function step(Synapse $synapse, float $gradient) : float;
}
