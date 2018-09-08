<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;

interface Output extends Parametric
{
    /**
     * Calculate the gradients for each output neuron and update.
     *
     * @param  array  $labels
     * @param  \Rubix\ML\NeuralNet\CostFunctions\CostFunction  $costFunction
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @throws \RuntimeException
     * @return array
     */
    public function back(array $labels, CostFunction $costFunction, Optimizer $optimizer) : array;
}
