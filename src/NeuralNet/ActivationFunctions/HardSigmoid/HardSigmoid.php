<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\HardSigmoid;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;

/**
 * HardSigmoid
 *
 * A piecewise linear approximation of the sigmoid function that is computationally
 * more efficient. The Hard Sigmoid function has an output value between 0 and 1,
 * making it useful for binary classification problems.
 *
 * f(x) = max(0, min(1, 0.2 * x + 0.5))
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class HardSigmoid implements ActivationFunction, IBufferDerivative
{
    /**
     * The slope of the linear region.
     *
     * @var float
     */
    protected const SLOPE = 0.2;

    /**
     * The y-intercept of the linear region.
     *
     * @var float
     */
    protected const INTERCEPT = 0.5;

    /**
     * The lower bound of the linear region.
     *
     * @var float
     */
    protected const LOWER_BOUND = -2.5;

    /**
     * The upper bound of the linear region.
     *
     * @var float
     */
    protected const UPPER_BOUND = 2.5;

    /**
     * Apply the HardSigmoid activation function to the input.
     *
     * f(x) = max(0, min(1, 0.2 * x + 0.5))
     *
     * @param NDArray $input The input values
     * @return NDArray The activated values
     */
    public function activate(NDArray $input) : NDArray
    {
        // Calculate 0.2 * x + 0.5
        $linear = NumPower::add(
            NumPower::multiply($input, self::SLOPE),
            self::INTERCEPT
        );

        // Clip values between 0 and 1
        return NumPower::clip($linear, 0.0, 1.0);
    }

    /**
     * Calculate the derivative of the activation function.
     *
     * f'(x) = 0.2 if -2.5 <= x <= 2.5
     * f'(x) = 0   otherwise
     *
     * @param NDArray $input Input matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $input) : NDArray
    {
        // For values in the linear region (-2.5 <= x <= 2.5): SLOPE
        $inLinearRegion = NumPower::greaterEqual($input, self::LOWER_BOUND);
        $inLinearRegion = NumPower::multiply($inLinearRegion, NumPower::lessEqual($input, self::UPPER_BOUND));
        $linearPart = NumPower::multiply($inLinearRegion, self::SLOPE);

        // For values outside the linear region: 0
        // Since we're multiplying by 0 for these regions, we don't need to explicitly handle them
        // The mask $inLinearRegion already contains 0s for x <= -2.5 and x >= 2.5,
        // so when we multiply by SLOPE, those values remain 0 in the result

        return $linearPart;
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'HardSigmoid';
    }
}
