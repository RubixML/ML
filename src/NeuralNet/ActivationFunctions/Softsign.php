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
     * @param Matrix $x
     * @return Matrix
     */
    public function activate(Matrix $x) : Matrix
    {
        return $x->map([$this, '_activate']);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * @internal
     *
     * @param Matrix $x
     * @param Matrix $z
     * @return Matrix
     */
    public function differentiate(Matrix $x, Matrix $z) : Matrix
    {
        return $x->map([$this, '_differentiate']);
    }

    /**
     * @internal
     *
     * @param float $x
     * @return float
     */
    public function _activate(float $x) : float
    {
        return $x / (1.0 + abs($x));
    }

    /**
     * @internal
     *
     * @param float $x
     * @return float
     */
    public function _differentiate(float $x) : float
    {
        return 1.0 / (1.0 + abs($x)) ** 2;
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
