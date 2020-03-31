<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Output extends Layer
{
    /**
     * Compute the gradient and loss at the output.
     *
     * @param (string|int|float)[] $labels
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return mixed[]
     */
    public function back(array $labels, Optimizer $optimizer) : array;
}
