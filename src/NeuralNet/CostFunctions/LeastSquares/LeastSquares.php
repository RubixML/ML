<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions\LeastSquares;

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
     * @internal
     *
     * @param NDArray $output The output of the network
     * @param NDArray $target The target values
     * @return float
     */
    public function compute(NDArray $output, NDArray $target) : float
    {
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
     * @internal
     *
     * @param NDArray $output The output of the network
     * @param NDArray $target The target values
     * @return NDArray
     */
    public function differentiate(NDArray $output, NDArray $target) : NDArray
    {
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
