<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Optimizers\Optimizer;

/**
 * Output
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Output extends Layer
{
    /**
     * Compute the gradient and loss at the output.
     *
     * @param (string|int|float)[] $labels
     * @param Optimizer $optimizer
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return mixed[]
     */
    public function back(array $labels, Optimizer $optimizer) : array;
}
