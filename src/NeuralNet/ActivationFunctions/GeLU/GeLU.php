<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\GeLU;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\SingleBufferDerivative;

/**
 * GeLU
 *
 * Gaussian Error Linear Units (GeLUs) are rectifiers that are gated by the magnitude of their input rather
 * than the sign of their input as with ReLU variants. Their output can be interpreted as the expected value
 * of a neuron with random dropout regularization applied.
 *
 * References:
 * [1] D. Hendrycks et al. (2018). Gaussian Error Linear Units (GeLUs).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class GeLU implements ActivationFunction, SingleBufferDerivative
{
    /**
     * The square root of two over pi constant sqrt(2/π).
     *
     * @var float
     */
    protected const ALPHA = 0.7978845608;

    /**
     * Gaussian error function approximation term.
     *
     * @var float
     */
    protected const BETA = 0.044715;

    /**
     * Apply the GeLU activation function to the input.
     *
     * f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
     *
     * @param NDArray $input The input values
     * @return NDArray The activated values
     */
    public function activate(NDArray $input) : NDArray
    {
        // Calculate x^3
        $cubed = $input ** 3;

        // Calculate inner term: x + BETA * x^3
        $innerTerm = NumPower::add(
            a: $input,
            b: NumPower::multiply(a: self::BETA, b: $cubed)
        );

        // Apply tanh(ALPHA * innerTerm)
        $tanhTerm = NumPower::tanh(
            NumPower::multiply(a: self::ALPHA, b: $innerTerm)
        );

        // Calculate 1 + tanhTerm
        $onePlusTanh = NumPower::add(a: 1.0, b: $tanhTerm);

        // Calculate 0.5 * x * (1 + tanhTerm)
        return NumPower::multiply(
            a: 0.5,
            b: NumPower::multiply(a: $input, b: $onePlusTanh)
        );
    }

    /**
     * Calculate the derivative of the activation function.
     *
     * The derivative of GeLU is:
     * f'(x) = 0.5 * (1 + tanh(α * (x + β * x^3))) +
     *         0.5 * x * sech^2(α * (x + β * x^3)) * α * (1 + 3β * x^2)
     *
     * Where:
     * - α = sqrt(2/π) ≈ 0.7978845608
     * - β = 0.044715
     * - sech^2(z) = (1/cosh(z))^2
     *
     * @param NDArray $x Output matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $x) : NDArray
    {
        // Calculate x^3
        $cubed = $x ** 3;

        // Calculate inner term: ALPHA * (x + BETA * x^3)
        $innerTerm = NumPower::multiply(
            a: self::ALPHA,
            b: NumPower::add(
                a: $x,
                b: NumPower::multiply(a: self::BETA, b: $cubed)
            )
        );

        // Calculate cosh and sech^2
        $cosh = NumPower::cosh($innerTerm);
        $sech2 = NumPower::pow(
            a: NumPower::divide(a: 1.0, b: $cosh),
            b: 2
        );

        // Calculate 0.5 * (1 + tanh(innerTerm))
        $firstTerm = NumPower::multiply(
            a: 0.5,
            b: NumPower::add(a: 1.0, b: NumPower::tanh($innerTerm))
        );

        // Calculate 0.5 * x * sech^2 * ALPHA * (1 + 3 * BETA * x^2)
        $secondTerm = NumPower::multiply(
            a: NumPower::multiply(
                a: NumPower::multiply(
                    a: 0.5 * self::ALPHA,
                    b: $x
                ),
                b: $sech2
            ),
            b: NumPower::add(
                a: 1.0,
                b: NumPower::multiply(
                    a: 3.0 * self::BETA,
                    b: NumPower::pow(a: $x, b: 2)
                )
            )
        );

        // Combine terms
        return NumPower::add(a: $firstTerm, b: $secondTerm);
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'GeLU';
    }
}
