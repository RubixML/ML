<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Tensor;
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
     * Return a tuple of the min and max output value for this function.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0., INF];
    }

    /**
     * Compute the loss score.
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
     * @param \Tensor\Tensor<int|float|array> $output
     * @param \Tensor\Tensor<int|float|array> $target
     * @return \Tensor\Tensor<int|float|array>
     */
    public function differentiate(Tensor $output, Tensor $target) : Tensor
    {
        return $output->subtract($target);
    }
}
