<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\OBufferDerivative;

/**
 * Sigmoid
 *
 * A bounded S-shaped function (sometimes called the *Logistic* function) with an output value
 * between 0 and 1. The output of the sigmoid function has the advantage of being interpretable
 * as a probability, however it is not zero-centered and tends to saturate if inputs become large.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Sigmoid implements ActivationFunction, OBufferDerivative
{
    /**
     * Compute the activation.
     *
     * f(x) = 1 / (1 + e^(-x))
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function activate(NDArray $input) : NDArray
    {
        $negExp = NumPower::exp(NumPower::multiply($input, -1.0));
        $denominator = NumPower::add(1.0, $negExp);

        return NumPower::divide(1.0, $denominator);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * For Sigmoid, the derivative can be calculated using only the output:
     * f'(x) = f(x) * (1 - f(x))
     * where f(x) is the output of the sigmoid function
     *
     * @param NDArray $output
     * @return NDArray
     */
    public function differentiate(NDArray $output) : NDArray
    {
        $oneMinusOutput = NumPower::subtract(1.0, $output);

        return NumPower::multiply($output, $oneMinusOutput);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Sigmoid';
    }
}
