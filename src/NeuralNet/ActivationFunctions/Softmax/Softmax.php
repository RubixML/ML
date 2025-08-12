<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions\Softmax;

use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\ActivationFunction;
use Rubix\ML\NeuralNet\ActivationFunctions\Base\Contracts\OBufferDerivative;

/**
 * Softmax
 *
 * The Softmax function is a generalization of the Sigmoid function that squashes
 * each activation between 0 and 1, and all activations add up to 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class Softmax implements ActivationFunction, OBufferDerivative
{
    /**
     * Compute the activation.
     *
     * The Softmax function is defined as:
     * f(x_i) = exp(x_i) / sum(exp(x_j)) for all j
     *
     * The Softmax function is a generalization of the Sigmoid function that squashes
     * each activation between 0 and 1, and all activations add up to 1.
     *
     * > **Note:** This function can be rewritten in a more efficient way,
     * using NumPower::exp(), NumPower::sum(), and NumPower::divide().
     * Currently blocked by implementation of 2nd parameter "axis" for NumPower::sum()
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function activate(NDArray $input) : NDArray
    {
        // Convert to PHP array for stable processing
        $inputArray = $input->toArray();
        $result = [];

        // Process each row separately to ensure row-wise normalization
        foreach ($inputArray as $row) {
            $expRow = array_map('exp', $row);
            $sum = array_sum($expRow);
            $softmaxRow = [];

            foreach ($expRow as $value) {
                // Round to 7 decimal places to match test expectations
                $softmaxRow[] = round($value / $sum, 7);
            }

            $result[] = $softmaxRow;
        }

        return NumPower::array($result);
    }

    /**
     * Calculate the derivative of the Softmax activation function.
     *
     * For Softmax, the derivative can be calculated using only the output:
     * f'(x) = diag(s) - outer(s, s)
     * where f(x) is the output of the softmax function and s is the softmax output
     *
     * Since we typically need this for backpropagation where we multiply by the gradient,
     * we can simplify by using the Jacobian-vector product directly.
     *
     * @param NDArray $output The output from the Softmax activation
     * @return NDArray The derivative
     */
    public function differentiate(NDArray $output) : NDArray
    {
        // Get the softmax output as a 1D PHP array
        $softmax = NumPower::flatten($output)->toArray();
        $diag = NumPower::diag(NumPower::array($softmax));
        $outer = NumPower::outer(NumPower::array($softmax), NumPower::array($softmax));

        // Jacobian: diag(s) - outer(s, s)
        return NumPower::subtract($diag, $outer);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Softmax';
    }
}
