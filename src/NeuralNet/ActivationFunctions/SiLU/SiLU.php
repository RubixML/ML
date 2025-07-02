<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\SiLU;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid\Sigmoid;

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
class SiLU implements ActivationFunction, IBufferDerivative
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
        // Calculate sigmoid(x) using the Sigmoid activation function
        $sigmoid = $this->sigmoid->activate($input);

        // Calculate x * sigmoid(x)
        return NumPower::multiply($input, $sigmoid);
    }

    /**
     * Calculate the derivative of the activation.
     *
     * f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
     *        = sigmoid(x) + x * sigmoid'(x)
     *
     * @param NDArray $input Input matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $input) : NDArray
    {
        // Calculate sigmoid(x) using the Sigmoid activation function
        $sigmoid = $this->sigmoid->activate($input);

        // Calculate sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        $sigmoidDerivative = $this->sigmoid->differentiate($sigmoid);

        // Calculate x * sigmoid'(x)
        $xTimesSigmoidDerivative = NumPower::multiply($input, $sigmoidDerivative);

        // Calculate sigmoid(x) + x * sigmoid'(x)
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
