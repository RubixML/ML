<?php

namespace Rubix\Engine\NeuralNetwork\Optimizers;

use Rubix\Engine\NeuralNetwork\Synapse;

interface Optimizer
{
    /**
     * Calculate the amount of a step of gradient descent.
     *
     * @param  \Rubix\Engine\NeuralNetwork\Synapse  $synapse
     * @param  float  $gradient
     * @return void
     */
    public function step(Synapse $synapse, float $gradient) : void;
}
