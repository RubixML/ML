<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use NumPower;
use NDArray;

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
        $denominator = NumPower::add(1.0, $absInput);

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
        $absInput = NumPower::abs($input);
        $onePlusAbs = NumPower::add(1.0, $absInput);
        $denominator = NumPower::multiply($onePlusAbs, $onePlusAbs);

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
