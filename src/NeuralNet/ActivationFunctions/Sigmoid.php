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
     * @param Matrix $input
     * @return Matrix
     */
    public function activate(Matrix $input) : Matrix
    {
        return $input->map('Rubix\ML\sigmoid');
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
     * @param float $output
     * @return float
     */
    public function _differentiate(float $output) : float
    {
        return $output * (1.0 - $output);
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
