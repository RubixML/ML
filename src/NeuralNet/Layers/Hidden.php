<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Hidden extends Layer
{
    /**
     * Calculate the errors and gradients of the layer and update the parameters.
     *
     * @param  callable  $prevErrors
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @return callable
     */
    public function back(callable $prevErrors, Optimizer $optimizer) : callable;
}
