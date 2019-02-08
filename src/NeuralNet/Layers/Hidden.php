<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Closure;

interface Hidden extends Layer
{
    /**
     * Calculate the gradients and update the parameters of the layer.
     *
     * @param Closure $prevGradient
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @return Closure
     */
    public function back(Closure $prevGradient, Optimizer $optimizer) : Closure;
}
