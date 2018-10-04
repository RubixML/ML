<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\Tensor\Matrix;

interface CostFunction
{
    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return array
     */
    public function range() : array;

    /**
     * Compute the cost.
     *
     * @param  \Rubix\Tensor\Matrix  $expected
     * @param  \Rubix\Tensor\Matrix  $activations
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $expected, Matrix $activations) : Matrix;

    /**
     * Calculate the derivatives of the cost function with respect to the
     * output activation.
     *
     * @param  \Rubix\Tensor\Matrix  $expected
     * @param  \Rubix\Tensor\Matrix  $activations
     * @param  \Rubix\Tensor\Matrix  $delta
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $activations, Matrix $delta) : Matrix;
}
