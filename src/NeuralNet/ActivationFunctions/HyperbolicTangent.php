<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;

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
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-1., 1.];
    }

    /**
     * Compute the output value.
     *
     * @param \Rubix\Tensor\Matrix $z
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map('tanh');
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param \Rubix\Tensor\Matrix $z
     * @param \Rubix\Tensor\Matrix $computed
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $computed->map([$this, '_differentiate']);
    }

    /**
     * @param float $computed
     * @return float
     */
    public function _differentiate(float $computed) : float
    {
        return 1. - ($computed ** 2);
    }
}
