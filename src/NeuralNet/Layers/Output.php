<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Output extends Parametric
{
    /**
     * Calculate the errors and gradients of the layer and update the parameters.
     *
     * @param  array  $labels
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @return array
     */
    public function back(array $labels, Optimizer $optimizer) : array;
}
