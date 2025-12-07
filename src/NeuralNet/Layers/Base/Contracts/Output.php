<?php

namespace Rubix\ML\NeuralNet\Layers\Base\Contracts;

use Rubix\ML\NeuralNet\Optimizers\Base\Optimizer;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Output
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface Output extends Layer
{
    /**
     * Compute the gradient and loss at the output.
     *
     * @param (string|int|float)[] $labels
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return mixed[]
     */
    public function back(array $labels, Optimizer $optimizer) : array;
}
