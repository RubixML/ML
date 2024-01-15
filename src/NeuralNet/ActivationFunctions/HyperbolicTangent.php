<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

/**
 * Hyperbolic Tangent
 *
 * S-shaped function that squeezes the input value into an output space between
 * -1 and 1 centered at 0.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class HyperbolicTangent implements ActivationFunction
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
        return $input->map('tanh');
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
        return 1.0 - ($output ** 2);
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
        return 'Hyperbolic Tangent';
    }
}
