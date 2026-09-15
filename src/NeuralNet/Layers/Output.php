<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;

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
     * @param Matrix $y
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return mixed[]
     */
    public function back(Matrix $y) : array;
}
