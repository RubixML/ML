<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;

/**
 * Gaussian
 *
 * The Gaussian activation function is a type of Radial Basis Function (*RBF*)
 * whose activation depends only on the distance the input is from the origin.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Gaussian implements ActivationFunction
{
    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0., 1.];
    }

    /**
     * Compute the output value.
     *
     * @param  \Rubix\Tensor\Matrix  $z
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map([$this, '_compute']);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param  \Rubix\Tensor\Matrix  $z
     * @param  \Rubix\Tensor\Matrix  $computed
     * @return \Rubix\Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $z->multiply($computed)->multiply(-2.);
    }

    /**
     * @param  float  $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return exp(-($z ** 2));
    }
}
