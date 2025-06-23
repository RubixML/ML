<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;

/**
 * Hyperbolic Tangent
 *
 * S-shaped function that squeezes the input value into an output space between
 * -1 and 1 centered at 0.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class HyperbolicTangent implements ActivationFunction, IBufferDerivative
{
    /**
     * Apply the Hyperbolic Tangent activation function to the input.
     *
     * f(x) = tanh(x)
     *
     * @param NDArray $input The input values
     * @return NDArray The activated values
     */
    public function activate(NDArray $input) : NDArray
    {
        return NumPower::tanh($input);
    }

    /**
     * Calculate the derivative of the activation function.
     *
     * f'(x) = 1 - tanh^2(x)
     *
     * @param NDArray $output Output matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $output) : NDArray
    {
        // Calculate tanh^2(x)
        $squared = NumPower::pow($output, 2);

        // Calculate 1 - tanh^2(x)
        return NumPower::subtract(1.0, $squared);
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'Hyperbolic Tangent';
    }
}
