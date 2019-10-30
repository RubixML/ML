<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

/**
 * SELU
 *
 * Scaled Exponential Linear Unit is a self-normalizing activation function
 * based on the ELU activation function. Neuronal activations of SELU networks
 * automatically converge toward zero mean and unit variance, unlike explicitly
 * normalized networks such as those with [Batch Norm](#batch-norm).
 *
 * References:
 * [1] G. Klambauer et al. (2017). Self-Normalizing Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SELU implements ActivationFunction
{
    public const ALPHA = 1.6732632423543772848170429916717;
    public const SCALE = 1.0507009873554804934193349852946;

    protected const BETA = self::SCALE * self::ALPHA;

    /**
     * Return a tuple of the min and max output value for this activation
     * function.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-self::BETA, INF];
    }

    /**
     * Compute the output value.
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
     * @param \Tensor\Matrix $z
     * @param \Tensor\Matrix $computed
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        return $computed->map([$this, '_differentiate']);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > 0. ? self::SCALE * $z : self::BETA * (exp($z) - 1.);
    }

    /**
     * @param float $computed
     * @return float
     */
    public function _differentiate(float $computed) : float
    {
        return $computed > 0. ? self::SCALE : self::SCALE * ($computed + self::ALPHA);
    }
}
