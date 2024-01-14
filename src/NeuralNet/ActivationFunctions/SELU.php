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
    /**
     * The value at which leakage starts to saturate.
     *
     * @var float
     */
    public const ALPHA = 1.6732632423543772848170429916717;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    public const SCALE = 1.0507009873554804934193349852946;

    /**
     * The scaling coefficient multiplied by alpha.
     *
     * @var float
     */
    protected const BETA = self::SCALE * self::ALPHA;

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
        return $input->map([$this, '_activate']);
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
     * @param float $input
     * @return float
     */
    public function _activate(float $input) : float
    {
        return $input > 0.0
            ? self::SCALE * $input
            : self::BETA * (exp($input) - 1.0);
    }

    /**
     * @internal
     *
     * @param float $output
     * @return float
     */
    public function _differentiate(float $output) : float
    {
        return $output > 0.0
            ? self::SCALE
            : self::SCALE * ($output + self::ALPHA);
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
