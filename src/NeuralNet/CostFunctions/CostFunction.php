<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Tensor;
use Tensor\Matrix;

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
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return float
     */
    public function compute(Matrix $output, Matrix $target) : float;

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \Tensor\Tensor<int|float|array> $output
     * @param \Tensor\Tensor<int|float|array> $target
     * @return \Tensor\Tensor<int|float|array>
     */
    public function differentiate(Tensor $output, Tensor $target) : Tensor;
}
