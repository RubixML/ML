<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use NumPower;
use NDArray;
use Rubix\ML\Exceptions\InvalidLeakageException;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;

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
class LeakyReLU implements ActivationFunction
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

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

        $this->leakage = $leakage;
    }

    /**
     * Apply the Leaky ReLU activation function to the input.
     *
     * f(x) = x           if x > 0
     * f(x) = leakage * x if x ≤ 0
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function activate(NDArray $input) : NDArray
    {
        $positiveActivation = NumPower::maximum($input, 0);

        $negativeActivation = NumPower::multiply(
            NumPower::minimum($input, 0),
            $this->leakage
        );

        return NumPower::add($positiveActivation, $negativeActivation);
    }

    /**
     * Calculate the derivative of the activation function.
     *
     * f'(x) = 1         if x > 0
     * f'(x) = leakage   if x ≤ 0
     *
     * @param NDArray $input
     * @param NDArray $output
     * @return NDArray
     */
    public function differentiate(NDArray $input, NDArray $output) : NDArray
    {
        $positivePart = NumPower::greater($input, 0);

        $negativePart = NumPower::multiply(
            NumPower::lessEqual($input, 0),
            $this->leakage
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
        return "Leaky ReLU (leakage: {$this->leakage})";
    }
}
