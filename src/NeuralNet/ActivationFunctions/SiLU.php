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
 * [1] S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function
 * Approximation in Reinforcement Learning.
 * [2] P. Ramachandran et al. (2017). Swish: A Self-gated Activation Function.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SiLU implements ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @return \Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        return $z->map([$this, '_compute']);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @internal
     *
     * @param \Tensor\Matrix $z
     * @param \Tensor\Matrix $computed
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        $ones = Matrix::ones(...$computed->shape());

        return $computed->divide($z)
            ->multiply($ones->subtract($computed))
            ->add($computed);
    }

    /**
     * @internal
     *
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z / (1.0 + exp(-$z));
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
