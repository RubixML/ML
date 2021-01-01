<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Matrix;

/**
 * Least Squares
 *
 * Least Squares or *quadratic* loss is a function that measures the squared
 * error between the target output and the actual output of a network.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeastSquares implements RegressionLoss
{
    /**
     * Compute the loss score.
     *
     * @internal
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return float
     */
    public function compute(Matrix $output, Matrix $target) : float
    {
        return $output->subtract($target)->square()->mean()->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @internal
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $output, Matrix $target) : Matrix
    {
        return $output->subtract($target);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Least Squares';
    }
}
