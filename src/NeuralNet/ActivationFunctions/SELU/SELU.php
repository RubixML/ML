<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\SELU;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;

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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class SELU implements ActivationFunction, IBufferDerivative
{
    /**
     * The value at which leakage starts to saturate.
     *
     * @var float
     */
    public const ALPHA = 1.6732632;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    public const LAMBDA = 1.0507009;

    /**
     * The scaling coefficient multiplied by alpha.
     *
     * @var float
     */
    protected const BETA = self::LAMBDA * self::ALPHA;

    /**
     * Compute the activation.
     *
     * f(x) = λ * x                 if x > 0
     * f(x) = λ * α * (e^x - 1)     if x ≤ 0
     *
     * @param NDArray $input The input values
     * @return NDArray The activated values
     */
    public function activate(NDArray $input) : NDArray
    {
        $positive = NumPower::multiply(
            NumPower::maximum($input, 0),
            self::LAMBDA
        );

        $negativeMask = NumPower::minimum($input, 0);
        $negative = NumPower::multiply(
            NumPower::expm1($negativeMask),
            self::BETA
        );

        return NumPower::add($positive, $negative);
    }

    /**
     * Calculate the derivative of the SELU activation function.
     *
     * f'(x) = λ                if x > 0
     * f'(x) = λ * α * e^x      if x ≤ 0
     *
     * @param NDArray $input Input matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $input) : NDArray
    {
        $positiveMask = NumPower::greater($input, 0);
        $positivePart = NumPower::multiply($positiveMask, self::LAMBDA);

        $negativeMask = NumPower::lessEqual($input, 0);
        $negativePart = NumPower::multiply(
            NumPower::multiply(
                NumPower::exp(
                    NumPower::multiply($negativeMask, $input)
                ),
                self::BETA
            ),
            $negativeMask
        );

        return NumPower::add($positivePart, $negativePart);
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'SELU';
    }
}
