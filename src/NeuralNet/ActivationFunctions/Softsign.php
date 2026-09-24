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
        return $x->divide($x->abs()->add(1.0));
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
        return $x->abs()->add(1.0)->pow(2.0)->reciprocal();
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
