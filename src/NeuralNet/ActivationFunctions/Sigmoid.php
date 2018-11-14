<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;

/**
 * Sigmoid
 *
 * A bounded S-shaped function (specifically the Logistic function) with an
 * output value between 0 and 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Sigmoid implements ActivationFunction
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
        return $computed->map([$this, '_differentiate']);
    }

    /**
     * @param  float  $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return 1. / (1. + exp(-$z));
    }

    /**
     * @param  float  $computed
     * @return float
     */
    public function _differentiate(float $computed) : float
    {
        return $computed * (1. - $computed);
    }
}
