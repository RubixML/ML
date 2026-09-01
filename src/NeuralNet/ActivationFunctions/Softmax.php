<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use NumPower;
use NDArray;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;

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
class Softmax implements ActivationFunction
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

        $maxima = [];

        for ($column = 0; $column < $columns; ++$column) {
            $maximum = -INF;

            foreach ($values as $row) {
                $maximum = max($maximum, $row[$column]);
            }

            $maxima[] = $maximum;
        }

        $maxima = NumPower::array($maxima, $input->dataType());

        $max = NumPower::reshape($maxima, [1, $columns]);

        $exponentials = NumPower::exp(NumPower::subtract($input, $max));

        $totals = NumPower::reshape(NumPower::sum($exponentials, axis: 0), [1, $columns]);

        return NumPower::divide($exponentials, $totals);
    }

    /**
     * Calculate the derivative of the Softmax activation function.
     *
     * Returns the element-wise diagonal of each sample's Softmax Jacobian:
     * f'(x_i) = f(x_i) * (1 - f(x_i))
     *
     * The result has the same shape as the input, preserving the
     * `[classes, batch]` layout where each sample column is treated
     * independently. Since the full Softmax Jacobian couples all classes of a
     * sample together, an exact backward pass through this function requires
     * the Jacobian-vector product which must be handled by the output layer.
     *
     * @param NDArray $input
     * @param NDArray $output The output from the Softmax activation
     * @return NDArray The derivative
     */
    public function differentiate(NDArray $input, NDArray $output) : NDArray
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
        return 'Softmax';
    }
}
