<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

/**
 * Softsign
 *
 * A function that squashes the output of a neuron to + or - 1 from 0. In other
 * words, the output is between -1 and 1.
 *
 * References:
 * [1] X. Glorot et al. (2010). Understanding the Difficulty of Training Deep
 * Feedforward Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Softsign implements ActivationFunction
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
        return $input->map([$this, '_differentiate']);
    }

    /**
     * @internal
     *
     * @param float $input
     * @return float
     */
    public function _activate(float $input) : float
    {
        return $input / (1.0 + abs($input));
    }

    /**
     * @internal
     *
     * @param float $input
     * @return float
     */
    public function _differentiate(float $input) : float
    {
        return 1.0 / (1.0 + abs($input)) ** 2;
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
        return 'Softsign';
    }
}
