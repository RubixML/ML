<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Tensor;

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
     * Compute the loss.
     *
     * @param \Rubix\Tensor\Tensor $expected
     * @param \Rubix\Tensor\Tensor $output
     * @return float
     */
    public function compute(Tensor $expected, Tensor $output) : float
    {
        return $output->subtract($expected)->square()->sum()->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \Rubix\Tensor\Tensor $expected
     * @param \Rubix\Tensor\Tensor $output
     * @return \Rubix\Tensor\Tensor
     */
    public function differentiate(Tensor $expected, Tensor $output) : Tensor
    {
        return $output->subtract($expected);
    }
}
