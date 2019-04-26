<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Tensor;

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
     * @param \Rubix\Tensor\Tensor $expected
     * @param \Rubix\Tensor\Tensor $output
     * @return float
     */
    public function compute(Tensor $expected, Tensor $output) : float;

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \Rubix\Tensor\Tensor $expected
     * @param \Rubix\Tensor\Tensor $output
     * @return \Rubix\Tensor\Tensor
     */
    public function differentiate(Tensor $expected, Tensor $output) : Tensor;
}
