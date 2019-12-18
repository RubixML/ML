<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Output extends Parametric
{
    /**
     * Calculate the gradients for each output neuron and update.
     *
     * @param (string|int|float)[] $labels
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return mixed[]
     */
    public function back(array $labels, Optimizer $optimizer) : array;
}
