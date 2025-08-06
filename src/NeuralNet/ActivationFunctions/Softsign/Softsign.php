<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\Softsign;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;

/**
 * Softsign
 *
 * A function that squashes the output of a neuron to + or - 1 from 0. In other
 * words, the output is between -1 and 1.
 *
 * References:
 * [1] X. Glorot et al. (2010). Understanding the Difficulty of Training Deep
 * Feedforward Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Softsign implements ActivationFunction, IBufferDerivative
{
    /**
     * Compute the activation.
     *
     * f(x) = x / (1 + |x|)
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function activate(NDArray $input) : NDArray
    {
        $absInput = NumPower::abs($input);

        // Calculate 1 + |x|
        $denominator = NumPower::add(1.0, $absInput);

        // Calculate x / (1 + |x|)
        return NumPower::divide($input, $denominator);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * f'(x) = 1 / (1 + |x|)²
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function differentiate(NDArray $input) : NDArray
    {
        // Calculate |x|
        $absInput = NumPower::abs($input);

        // Calculate 1 + |x|
        $onePlusAbs = NumPower::add(1.0, $absInput);

        // Calculate (1 + |x|)²
        $denominator = NumPower::multiply($onePlusAbs, $onePlusAbs);

        // Calculate 1 / (1 + |x|)²
        return NumPower::divide(1.0, $denominator);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Softsign';
    }
}
