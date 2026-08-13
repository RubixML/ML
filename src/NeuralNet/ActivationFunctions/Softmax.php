<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use NumPower;
use NDArray;

/**
 * Softmax
 *
 * The Softmax function is a generalization of the Sigmoid function that squashes
 * each activation between 0 and 1, and all activations add up to 1.
 *
 * Expects network layout `[classes, batch]` and normalizes each sample column.
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
     * Numerically stable form subtracts the per-sample max before exponentiation.
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function activate(NDArray $input) : NDArray
    {
        $columns = $input->shape()[1];
        $values = $input->toArray();

        // NumPower::max() has no axis argument, so compute column maxima in PHP.
        $maxima = [];

        for ($column = 0; $column < $columns; ++$column) {
            $maximum = -INF;

            foreach ($values as $row) {
                $maximum = max($maximum, $row[$column]);
            }

            $maxima[] = $maximum;
        }

        $max = NumPower::reshape(NumPower::array($maxima), [1, $columns]);
        $exponentials = NumPower::exp(NumPower::subtract($input, $max));
        $totals = NumPower::reshape(NumPower::sum($exponentials, axis: 0), [1, $columns]);

        return NumPower::divide($exponentials, $totals);
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
