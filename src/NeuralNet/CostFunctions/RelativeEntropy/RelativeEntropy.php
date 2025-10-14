<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions\RelativeEntropy;

use NumPower;
use NDArray;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\CostFunctions\Base\Contracts\ClassificationLoss;

use const Rubix\ML\EPSILON;

/**
 * Relative Entropy
 *
 * Relative Entropy or *Kullback-Leibler divergence* is a measure of how the
 * expectation and activation of the network diverge.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class RelativeEntropy implements ClassificationLoss
{
    /**
     * Compute the loss.
     *
     * L(y, ŷ) = Σ(y * log(y / ŷ)) / n
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

        // Clip values to avoid log(0)
        $target = NumPower::clip($target, EPSILON, 1.0);
        $output = NumPower::clip($output, EPSILON, 1.0);

        $ratio = NumPower::divide($target, $output);
        $logRatio = NumPower::log($ratio);
        $product = NumPower::multiply($target, $logRatio);

        return NumPower::mean($product);
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * ∂L/∂ŷ = (ŷ - y) / ŷ
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

        // Clip values to avoid division by zero
        $target = NumPower::clip($target, EPSILON, 1.0);
        $output = NumPower::clip($output, EPSILON, 1.0);

        $diff = NumPower::subtract($output, $target);

        return NumPower::divide($diff, $output);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Relative Entropy';
    }
}
