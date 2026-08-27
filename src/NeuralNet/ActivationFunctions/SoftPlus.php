<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use NumPower;
use NDArray;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;

/**
 * SoftPlus
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
class SoftPlus implements ActivationFunction
{
    public function __construct()
    {
        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();
    }

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

        return NumPower::log1p($exp);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * f'(x) = 1 / (1 + e^(-x))
     *
     * @param NDArray $input
     * @param NDArray $output
     * @return NDArray
     */
    public function differentiate(NDArray $input, NDArray $output) : NDArray
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
        return 'SoftPlus';
    }
}
