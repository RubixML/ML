<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\ReLU6;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\IBufferDerivative;

/**
 * ReLU6
 *
 * ReLU6 is a variant of the Rectified Linear Unit that caps the maximum
 * activation at 6. This helps with quantized networks and promotes
 * sparsity in the activations.
 *
 * References:
 * [1] A. Howard et al. (2017). MobileNets: Efficient Convolutional Neural
 * Networks for Mobile Vision Applications.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class ReLU6 implements ActivationFunction, IBufferDerivative
{
    /**
     * Compute the activation.
     *
     * f(x) = min(max(0, x), 6)
     *
     * @param NDArray $input The input values
     * @return NDArray The activated values
     */
    public function activate(NDArray $input) : NDArray
    {
        // First apply ReLU: max(0, x)
        $reluActivation = NumPower::maximum($input, 0.0);
        
        // Then cap at 6: min(relu(x), 6)
        return NumPower::minimum($reluActivation, 6.0);
    }

    /**
     * Calculate the derivative of the activation function.
     *
     * f'(x) = 1 if 0 < x < 6, else 0
     *
     * @param NDArray $input Input matrix
     * @return NDArray Derivative matrix
     */
    public function differentiate(NDArray $input) : NDArray
    {
        // 1 where 0 < x < 6, 0 elsewhere
        $greaterThanZero = NumPower::greater($input, 0.0);
        $lessThanSix = NumPower::less($input, 6.0);
        
        // Combine conditions with logical AND
        return NumPower::multiply($greaterThanZero, $lessThanSix);
    }

    /**
     * Return the string representation of the activation function.
     *
     * @return string String representation
     */
    public function __toString() : string
    {
        return 'ReLU6';
    }
}
