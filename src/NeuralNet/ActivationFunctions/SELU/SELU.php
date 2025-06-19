<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions\SELU;

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
        $positive = NumPower::maximum($input, 0) * self::SCALE;
        $negative = self::BETA * NumPower::expm1($input);

        return $negative + $positive;
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
        $positive = NumPower::greater($output, 0) * self::SCALE;
        $negative = NumPower::lessEqual($output) * ($output + self::ALPHA) * self::SCALE;

        return $positive + $negative;
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
