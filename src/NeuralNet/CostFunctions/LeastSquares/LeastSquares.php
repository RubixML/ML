<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions\LeastSquares;

use InvalidArgumentException;
use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\CostFunctions\Base\Contracts\RegressionLoss;

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
     * @param NDArray $output The output of the network
     * @param NDArray $target The target values
     * @return float
     */
    public function compute(NDArray $output, NDArray $target) : float
    {
        if ($output->shape() !== $target->shape()) {
            throw new InvalidArgumentException('Output and target must have the same shape.');
        }

        $difference = NumPower::subtract($output, $target);
        $squared = NumPower::pow($difference, 2);

        // Compute mean of all elements
        return NumPower::mean($squared);
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * ∂L/∂ŷ = y - ŷ
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

        return NumPower::subtract($output, $target);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Least Squares';
    }
}
