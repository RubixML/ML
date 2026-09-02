<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use NumPower;
use NDArray;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;

/**
 * SiLU
 *
 * Sigmoid Linear Units are smooth and non-monotonic rectified activation functions. Their inputs are weighted by
 * the [Sigmoid](sigmoid.md) activation function acting as a self-gating mechanism.
 *
 * References:
 * [1] S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in
 * Reinforcement Learning.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class SiLU implements ActivationFunction
{
    /**
     * The Sigmoid activation function.
     *
     * @var Sigmoid
     */
    protected Sigmoid $sigmoid;

    /**
     * Class constructor.
     */
    public function __construct()
    {
        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

        $this->sigmoid = new Sigmoid();
    }

    /**
     * Compute the activation.
     *
     * f(x) = x * sigmoid(x) = x / (1 + e^(-x))
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function activate(NDArray $input) : NDArray
    {
        $sigmoid = $this->sigmoid->activate($input);

        return NumPower::multiply($input, $sigmoid);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
     *        = sigmoid(x) + x * sigmoid'(x)
     *
     * @param NDArray $input
     * @param NDArray $output
     * @return NDArray
     */
    public function differentiate(NDArray $input, NDArray $output) : NDArray
    {
        $sigmoid = $this->sigmoid->activate($input);
        $sigmoidDerivative = $this->sigmoid->differentiate($input, $sigmoid);
        $xTimesSigmoidDerivative = NumPower::multiply($input, $sigmoidDerivative);

        return NumPower::add($sigmoid, $xTimesSigmoidDerivative);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'SiLU';
    }
}
