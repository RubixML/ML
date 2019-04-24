<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;

interface CostFunction
{
    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return float[]
     */
    public function range() : array;

    /**
     * Compute the loss.
     *
     * @param \Rubix\Tensor\Matrix $expected
     * @param \Rubix\Tensor\Matrix $output
     * @return float
     */
    public function compute(Matrix $expected, Matrix $output) : float;

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \Rubix\Tensor\Matrix $expected
     * @param \Rubix\Tensor\Matrix $output
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $output) : Matrix;
}
