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
     * Compute the output value.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @return \Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map([$this, '_compute']);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @param \Tensor\Matrix $computed
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $z->map([$this, '_differentiate']);
    }

    /**
     * @internal
     *
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z / (1.0 + abs($z));
    }

    /**
     * @internal
     *
     * @param float $z
     * @return float
     */
    public function _differentiate(float $z) : float
    {
        return 1.0 / (1.0 + abs($z)) ** 2;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Softsign';
    }
}
