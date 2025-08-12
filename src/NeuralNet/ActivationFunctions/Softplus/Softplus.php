<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\Softplus;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;

/**
 * Soft Plus
 *
 * A smooth approximation of the ReLU function whose output is constrained to be
 * positive.
 *
 * References:
 * [1] X. Glorot et al. (2011). Deep Sparse Rectifier Neural Networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Softplus implements ActivationFunction, IBufferDerivative
{
    /**
     * Compute the activation.
     *
     * f(x) = log(1 + e^x)
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function activate(NDArray $input) : NDArray
    {
        $exp = NumPower::exp($input);
        $onePlusExp = NumPower::add(1.0, $exp);

        return NumPower::log($onePlusExp);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * f'(x) = 1 / (1 + e^(-x))
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function differentiate(NDArray $input) : NDArray
    {
        $negExp = NumPower::exp(NumPower::multiply($input, -1.0));
        $denominator = NumPower::add(1.0, $negExp);

        return NumPower::divide(1.0, $denominator);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Soft Plus';
    }
}
