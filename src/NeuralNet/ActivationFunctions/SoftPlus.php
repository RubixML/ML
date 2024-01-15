<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

/**
 * Soft Plus
 *
 * A smooth approximation of the ReLU function whose output is constrained to be
 * positive.
 *
 * References:
 * [1] X. Glorot et al. (2011). Deep Sparse Rectifier Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SoftPlus implements ActivationFunction
{
    /**
     * Compute the activation.
     *
     * @internal
     *
     * @param Matrix $input
     * @return Matrix
     */
    public function activate(Matrix $input) : Matrix
    {
        return $input->map([$this, '_activate']);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * @internal
     *
     * @param Matrix $input
     * @param Matrix $output
     * @return Matrix
     */
    public function differentiate(Matrix $input, Matrix $output) : Matrix
    {
        return $output->map([$this, '_differentiate']);
    }

    /**
     * @internal
     *
     * @param float $input
     * @return float
     */
    public function _activate(float $input) : float
    {
        return log(1.0 + exp($input));
    }

    /**
     * @internal
     *
     * @param float $output
     * @return float
     */
    public function _differentiate(float $output) : float
    {
        return 1.0 / (1.0 + exp(-$output));
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Soft Plus';
    }
}
