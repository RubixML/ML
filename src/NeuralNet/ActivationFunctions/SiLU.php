<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

use function exp;

/**
 * SiLU
 *
 * Sigmoid Linear Units are smooth and non-monotonic rectified activation functions. Their inputs are weighted by
 * the [Sigmoid](sigmoid.md) activation function acting as a self-gating mechanism.
 *
 * References:
 * [1] S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in
 * Reinforcement Learning.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SiLU implements ActivationFunction
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
        return $x->map([$this, '_compute']);
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
    public function _differentiate(float $x) : float
    {
        $sigmoid = 1.0 / (1.0 + exp(-$x));

        return $sigmoid + $x * $sigmoid * (1.0 - $sigmoid);
    }

    /**
     * @internal
     *
     * @param float $x
     * @return float
     */
    public function _compute(float $x) : float
    {
        return $x / (1.0 + exp(-$x));
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'SiLU';
    }
}
