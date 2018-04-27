<?php

namespace Rubix\Engine\NeuralNet\Optimizers;

use Rubix\Engine\NeuralNet\Synapse;

interface Optimizer
{
    const EPSILON = 1e-8;

    /**
     * Calculate the amount of a step of gradient descent.
     *
     * @param  \Rubix\Engine\NeuralNet\Synapse  $synapse
     * @param  float  $gradient
     * @return float
     */
    public function step(Synapse $synapse, float $gradient) : float;
}
