<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;

/**
 * ReLU
 *
 * ReLU (Rectified Linear Unit) is an activation function that only outputs
 * the positive signal of the input.
 *
 * References:
 * [1] A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural
 * Network Acoustic Models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ReLU implements ActivationFunction
{
    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [0, INF];
    }

    /**
     * Compute the output value.
     *
     * @param \Rubix\Tensor\Matrix $z
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map([$this, '_compute']);
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
        return $z->greater(0.);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > 0. ? $z : 0.;
    }
}
