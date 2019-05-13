<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Hidden extends Nonparametric
{
    /**
     * Calculate the gradient and update the parameters of the layer.
     *
     * @param \Rubix\ML\Deferred $prevGradient
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @return \Rubix\ML\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred;
}
