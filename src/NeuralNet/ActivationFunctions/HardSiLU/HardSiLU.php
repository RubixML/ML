<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\HardSiLU;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;
use Rubix\ML\NeuralNet\ActivationFunctions\HardSigmoid\HardSigmoid;

/**
 * HardSiLU
 *
 * Hard Sigmoid Linear Units (Hard SiLU) are a computationally efficient variant of the SiLU activation function.
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
class HardSiLU implements ActivationFunction, IBufferDerivative
{
    /**
     * The Hard Sigmoid activation function.
     *
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\HardSigmoid\HardSigmoid
     */
    protected HardSigmoid $hardSigmoid;

    /**
     * Class constructor.
     */
    public function __construct()
    {
        $this->hardSigmoid = new HardSigmoid();
    }

    /**
     * Apply the HardSiLU activation function to the input.
     *
     * f(x) = x * HardSigmoid(x)
     *
     * @param NDArray $input The input values
     * @return NDArray The activated values
     */
    public function activate(NDArray $input) : NDArray
    {
        // Calculate HardSigmoid(x)
        $hardSigmoid = $this->hardSigmoid->activate($input);

        // Calculate x * HardSigmoid(x)
        return NumPower::multiply($input, $hardSigmoid);
    }

    /**
     * Calculate the derivative of the activation function.
     *
     * f'(x) = HardSigmoid(x) + x * HardSigmoid'(x)
     *
     * @param NDArray $x Input matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $x) : NDArray
    {
        // Calculate HardSigmoid(x)
        $hardSigmoid = $this->hardSigmoid->activate($x);

        // Calculate HardSigmoid'(x)
        $hardSigmoidDerivative = $this->hardSigmoid->differentiate($x);

        // Calculate x * HardSigmoid'(x)
        $xTimesDerivative = NumPower::multiply($x, $hardSigmoidDerivative);

        // Calculate HardSigmoid(x) + x * HardSigmoid'(x)
        return NumPower::add($hardSigmoid, $xTimesDerivative);
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'HardSiLU';
    }
}
