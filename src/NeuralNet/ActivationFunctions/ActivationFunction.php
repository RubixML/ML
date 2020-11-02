<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

/**
 * Activation Function
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @return \Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix;

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @param \Tensor\Matrix $computed
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix;

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
