<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Hidden extends Layer
{
    /**
     * Calculate the gradients and update the parameters of the layer.
     *
     * @param  callable  $prevGradient
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @return callable
     */
    public function back(callable $prevGradient, Optimizer $optimizer) : callable;
}
