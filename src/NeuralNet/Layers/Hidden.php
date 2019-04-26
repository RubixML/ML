<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Hidden extends Nonparametric
{
    /**
     * Calculate the gradient and update the parameters of the layer.
     *
     * @param \Rubix\ML\NeuralNet\Deferred $prevGradient
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @return \Rubix\ML\NeuralNet\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred;
}
