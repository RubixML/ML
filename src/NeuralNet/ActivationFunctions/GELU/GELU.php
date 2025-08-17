<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\GELU;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;

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
class GELU implements ActivationFunction, IBufferDerivative
{
    /**
     * The square root of two over pi constant sqrt(2/π).
     *
     * @var float
     */
    protected const ALPHA = 0.7978845608;
    /** @var float 0.5 * ALPHA */
    protected const HALF_ALPHA = 0.3989422804;

    /**
     * Gaussian error function approximation term.
     *
     * @var float
     */
    protected const BETA = 0.044715;
    /** @var float 3 * BETA */
    protected const TRIPLE_BETA = 0.134145;

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
        $cubed = NumPower::pow($input, 3);
        $innerTerm = NumPower::add($input, NumPower::multiply($cubed, self::BETA));
        $tanhTerm = NumPower::tanh(NumPower::multiply($innerTerm, self::ALPHA));
        $onePlusTanh = NumPower::add(1.0, $tanhTerm);

        return NumPower::multiply(
            NumPower::multiply($input, $onePlusTanh),
            0.5
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
     * @param NDArray $input Input matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $input) : NDArray
    {
        $cubed = NumPower::pow($input, 3);

        $innerTerm = NumPower::multiply(
            NumPower::add(
                $input,
                NumPower::multiply($cubed, self::BETA)
            ),
            self::ALPHA
        );

        $cosh = NumPower::cosh($innerTerm);
        $sech2 = NumPower::pow(
            NumPower::divide(1.0, $cosh),
            2
        );

        $firstTerm = NumPower::multiply(
            NumPower::add(1.0, NumPower::tanh($innerTerm)),
            0.5
        );

        $secondTerm = NumPower::multiply(
            NumPower::multiply(
                NumPower::multiply(
                    $input,
                    self::HALF_ALPHA
                ),
                $sech2
            ),
            NumPower::add(
                1.0,
                NumPower::multiply(
                    NumPower::pow($input, 2),
                    self::TRIPLE_BETA
                )
            )
        );

        return NumPower::add($firstTerm, $secondTerm);
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'GELU';
    }
}
