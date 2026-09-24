<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

/**
 * Sigmoid
 *
 * A bounded S-shaped function (sometimes called the *Logistic* function) with an output value
 * between 0 and 1. The output of the sigmoid function has the advantage of being interpretable
 * as a probability, however it is not zero-centered and tends to saturate if inputs become large.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Sigmoid implements ActivationFunction
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
        return $x->negate()->exp()->add(1.0)->reciprocal();
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
        return $z->multiply($z->negate()->add(1.0));
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
        return 'Sigmoid';
    }
}
