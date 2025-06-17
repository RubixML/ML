<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\ELU;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\SingleBufferDerivative;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU\Exceptions\InvalidalphaException;

/**
 * ELU
 *
 * Exponential Linear Units are a type of rectifier that soften the transition
 * from non-activated to activated using the exponential function.
 *
 * References:
 * [1] D. A. Clevert et al. (2016). Fast and Accurate Deep Network Learning by
 * Exponential Linear Units.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Aleksei Nechaev <omfg.rus@gmail.com>
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class ELU implements ActivationFunction, SingleBufferDerivative
{
    /**
     * Class constructor.
     *
     * @param float $alpha At which negative value the ELU will saturate. For example if alpha
     *              equals 1, the leaked value will never be greater than -1.0.
     *
     * @throws InvalidalphaException
     */
    public function __construct(protected float $alpha = 1.0)
    {
        if ($this->alpha < 0.0) {
            throw new InvalidAlphaException(
                message: "Alpha must be greater than 0, $alpha given."
            );
        }
    }

    /**
     * Apply the ELU activation function to the input.
     *
     * f(x) = x                 if x > 0
     * f(x) = α * (e^x - 1)     if x ≤ 0
     *
     * @param NDArray $input The input values
     * @return NDArray The activated values
     */
    public function activate(NDArray $input) : NDArray
    {
        // Calculate positive part: x for x > 0
        $positiveActivation = NumPower::maximum(a: $input, b: 0);

        // Calculate negative part: alpha * (e^x - 1) for x <= 0
        $negativeMask = NumPower::minimum(a: $input, b: 0);
        $negativeActivation = NumPower::multiply(
            a: NumPower::expm1($negativeMask),
            b: $this->alpha
        );

        // Combine both parts
        return NumPower::add(a: $positiveActivation, b: $negativeActivation);
    }

    /**
     * Calculate the derivative of the activation function.
     *
     * f'(x) = 1               if x > 0
     * f'(x) = α * e^x         if x ≤ 0
     *
     * @param NDArray $x Output matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $x) : NDArray
    {
        // For x > 0: 1
        $positivePart = NumPower::greater(a: $x, b: 0);

        // For x <= 0: α * e^x
        $negativeMask = NumPower::lessEqual(a: $x, b: 0);
        $negativePart = NumPower::multiply(
            a: NumPower::multiply(a: $negativeMask, b: NumPower::exp($x)),
            b: $this->alpha
        );

        // Combine both parts
        return NumPower::add(a: $positivePart, b: $negativePart);
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return "ELU (alpha: {$this->alpha})";
    }
}
