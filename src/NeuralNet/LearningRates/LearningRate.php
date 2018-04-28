<?php

namespace Rubix\Engine\NeuralNet\LearningRates;

use Rubix\Engine\NeuralNet\Synapse;

interface LearningRate
{
    const EPSILON = 1e-8;

    /**
     * Calculate the value of a single step of gradient descent for a given
     * parameter.
     *
     * @param  \Rubix\Engine\NeuralNet\Synapse  $synapse
     * @param  float  $gradient
     * @return float
     */
    public function step(Synapse $synapse, float $gradient) : float;
}
