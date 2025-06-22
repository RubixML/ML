<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU\Exceptions\InvalidLeakageException;

/**
 * Leaky ReLU
 *
 * Leaky Rectified Linear Units are functions that output x when x > 0 or a
 * small leakage value when x < 0. The amount of leakage is controlled by the
 * user-specified parameter.
 *
 * References:
 * [1] A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural Network
 * Acoustic Models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class LeakyReLU implements ActivationFunction, IBufferDerivative
{
    /**
     * The amount of leakage as a ratio of the input value to allow to pass through when inactivated.
     *
     * @var float
     */
    protected float $leakage;

    /**
     * Class constructor.
     *
     * @param float $leakage The amount of leakage as a ratio of the input value to allow to pass through when inactivated.
     * @throws InvalidLeakageException
     */
    public function __construct(float $leakage = 0.1)
    {
        if ($leakage <= 0.0 || $leakage >= 1.0) {
            throw new InvalidLeakageException(
                message: "Leakage must be between 0 and 1, $leakage given."
            );
        }

        $this->leakage = $leakage;
    }

    /**
     * Apply the Leaky ReLU activation function to the input.
     *
     * f(x) = x           if x > 0
     * f(x) = leakage * x if x ≤ 0
     *
     * @param NDArray $input The input values
     * @return NDArray The activated values
     */
    public function activate(NDArray $input) : NDArray
    {
        // Calculate positive part: x for x > 0
        $positive = NumPower::maximum($input, 0);

        // Calculate negative part: leakage * x for x <= 0
        $negative = NumPower::multiply(
            NumPower::minimum($input, 0),
            $this->leakage
        );

        // Combine both parts
        return NumPower::add($positive, $negative);
    }

    /**
     * Calculate the derivative of the activation function.
     *
     * f'(x) = 1         if x > 0
     * f'(x) = leakage   if x ≤ 0
     *
     * @param NDArray $x Input matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $x) : NDArray
    {
        // For x > 0: 1
        $positive = NumPower::greater($x, 0);

        // For x <= 0: leakage
        $negative = NumPower::multiply(
            NumPower::lessEqual($x, 0),
            $this->leakage
        );

        // Combine both parts
        return NumPower::add($positive, $negative);
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return "Leaky ReLU (leakage: {$this->leakage})";
    }
}
