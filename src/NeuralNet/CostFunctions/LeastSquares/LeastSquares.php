<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions\LeastSquares;

use InvalidArgumentException;
use NumPower;
use NDArray;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares\Base\Contracts\RegressionLoss;

/**
 * Least Squares
 *
 * Least Squares or *quadratic* loss is a function that measures the squared
 * error between the target output and the actual output of a network.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class LeastSquares implements RegressionLoss
{
    /**
     * Compute the loss score.
     *
     * L(y, ŷ) = Σ(y - ŷ)^2 / n
     *
     * @internal
     *
     * @param NDArray $output The output of the network
     * @param NDArray $target The target values
     * @return float
     */
    public function compute(NDArray $output, NDArray $target) : float
    {
        if ($output->shape() !== $target->shape()) {
            throw new InvalidArgumentException('Output and target must have the same shape.');
        }

        // Compute difference: output - target
        $diff = NumPower::subtract($output, $target);

        // Square the difference: diff^2
        $squared = NumPower::pow($diff, 2);

        // Compute mean of all elements
        return NumPower::mean($squared);
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * ∂L/∂ŷ = y - ŷ
     *
     * @internal
     *
     * @param NDArray $output The output of the network
     * @param NDArray $target The target values
     * @return NDArray
     */
    public function differentiate(NDArray $output, NDArray $target) : NDArray
    {
        if ($output->shape() !== $target->shape()) {
            throw new InvalidArgumentException('Output and target must have the same shape.');
        }

        // Gradient is simply: output - target
        return NumPower::subtract($output, $target);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Least Squares';
    }
}
