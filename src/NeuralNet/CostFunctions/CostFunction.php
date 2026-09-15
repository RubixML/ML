<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Stringable;

/**
 * Cost Function
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface CostFunction extends Stringable
{
    /**
     * Compute the loss score.
     *
     * @internal
     *
     * @param Matrix $z
     * @param Matrix $y
     * @return float
     */
    public function compute(Matrix $z, Matrix $y) : float;

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @internal
     *
     * @param Matrix $z
     * @param Matrix $y
     * @return Matrix
     */
    public function differentiate(Matrix $z, Matrix $y) : Matrix;
}
