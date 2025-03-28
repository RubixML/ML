<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use NumPower;
use NDArray;

/**
 * Hyperbolic Tangent
 *
 * S-shaped function that squeezes the input value into an output space between
 * -1 and 1 centered at 0.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class HyperbolicTangent implements ActivationFunction
{
    /**
     * @inheritdoc
     */
    public function activate(NDArray $input) : NDArray
    {
        return NumPower::tanh($input);
    }

    /**
     * @inheritdoc
     */
    public function differentiate(NDArray $output) : NDArray
    {
        return 1 - ($output ** 2);
    }

    /**
     * Return the string representation of the activation function.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Hyperbolic Tangent';
    }
}
