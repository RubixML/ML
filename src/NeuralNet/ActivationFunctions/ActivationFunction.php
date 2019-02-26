<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;

interface ActivationFunction
{
    public const EPSILON = 1e-8;

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return float[]
     */
    public function range() : array;

    /**
     * Compute the output value.
     *
     * @param \Rubix\Tensor\Matrix $z
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix;

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param \Rubix\Tensor\Matrix $z
     * @param \Rubix\Tensor\Matrix $computed
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix;
}
