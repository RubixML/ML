<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Tensor\Matrix;

use function expm1;

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
    /**
     * The value at which leakage starts to saturate.
     *
     * @var float
     */
    public const float ALPHA = 1.6732632423543772848170429916717;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    public const float SCALE = 1.0507009873554804934193349852946;

    /**
     * The scaling coefficient multiplied by alpha.
     *
     * @var float
     */
    protected const float BETA = self::SCALE * self::ALPHA;

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
        return $x->map([$this, '_activate']);
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
        return $z->map([$this, '_differentiate']);
    }

    /**
     * @internal
     *
     * @param float $x
     * @return float
     */
    public function _activate(float $x) : float
    {
        return $x > 0.0 ? self::SCALE * $x : self::BETA * expm1($x);
    }

    /**
     * @internal
     *
     * @param float $z
     * @return float
     */
    public function _differentiate(float $z) : float
    {
        return $z > 0.0 ? self::SCALE : $z + self::BETA;
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
        return 'SELU';
    }
}
