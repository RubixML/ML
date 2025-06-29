<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\ReLU;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;

/**
 * ReLU
 *
 * ReLU (Rectified Linear Unit) is an activation function that only outputs
 * the positive signal of the input.
 *
 * References:
 * [1] A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural
 * Network Acoustic Models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class ReLU implements ActivationFunction, IBufferDerivative
{
    /**
     * Compute the activation.
     *
     * f(x) = max(0, x)
     *
     * @param NDArray $input The input values
     * @return NDArray The activated values
     */
    public function activate(NDArray $input) : NDArray
    {
        return NumPower::maximum($input, 0.0);
    }

    /**
     * Calculate the derivative of the activation function.
     *
     * f'(x) = 1 if x > 0, else 0
     *
     * @param NDArray $input Input matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $input) : NDArray
    {
        return NumPower::greater($input, 0.0);
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'ReLU';
    }
}
