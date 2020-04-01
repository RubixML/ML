<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Matrix;

interface CostFunction
{
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
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $output, Matrix $target) : Matrix;
}
