<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\Backends\Deferred;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Hidden extends Nonparametric
{
    /**
     * Calculate the gradient and update the parameters of the layer.
     *
     * @param \Rubix\ML\Backends\Deferred $prevGradient
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @return \Rubix\ML\Backends\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred;
}
