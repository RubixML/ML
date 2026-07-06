<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use Rubix\ML\Traits\AssertsShapes;
use const Rubix\ML\EPSILON;

/**
 * Relative Entropy
 *
 * Relative Entropy or *Kullback-Leibler divergence* is a measure of how the
 * expectation and activation of the network diverge.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class RelativeEntropy implements ClassificationLoss
{
    use AssertsShapes;

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
        $this->assertSameShape($output, $target);

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
        $this->assertSameShape($output, $target);

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
