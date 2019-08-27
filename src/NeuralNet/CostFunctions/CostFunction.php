<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Tensor;
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
     * Compute the loss score.
     *
     * @param \Rubix\Tensor\Matrix $output
     * @param \Rubix\Tensor\Matrix $target
     * @return float
     */
    public function compute(Matrix $output, Matrix $target) : float;

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \Rubix\Tensor\Tensor $output
     * @param \Rubix\Tensor\Tensor $target
     * @return \Rubix\Tensor\Tensor
     */
    public function differentiate(Tensor $output, Tensor $target) : Tensor;
}
