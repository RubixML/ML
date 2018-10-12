<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;

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
class LeastSquares implements CostFunction
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
     * Compute the loss matrix.
     *
     * @param  \Rubix\Tensor\Matrix  $expected
     * @param  \Rubix\Tensor\Matrix  $activations
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $expected, Matrix $activations) : Matrix
    {
        return $activations->subtract($expected)->square();
    }

    /**
     * Calculate the gradient of the cost function with respect to the
     * activation.
     *
     * @param  \Rubix\Tensor\Matrix  $expected
     * @param  \Rubix\Tensor\Matrix  $activations
     * @param  \Rubix\Tensor\Matrix  $delta
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $activations, Matrix $delta) : Matrix
    {
        return $activations->subtract($expected);
    }
}
