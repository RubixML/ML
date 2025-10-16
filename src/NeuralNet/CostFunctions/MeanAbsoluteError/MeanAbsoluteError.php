<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions\MeanAbsoluteError;

use InvalidArgumentException;
use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\CostFunctions\Base\Contracts\RegressionLoss;

/**
 * Mean Absolute Error
 *
 * Mean Absolute Error (MAE) measures the average magnitude of errors between
 * predicted and actual values without considering their direction. It is a
 * linear score which means all individual differences are weighted equally.
 * MAE is more robust to outliers compared to Mean Squared Error.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class MeanAbsoluteError implements RegressionLoss
{
    /**
     * Compute the loss score.
     *
     * L(y, ŷ) = Σ|y - ŷ| / n
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
        $absolute = NumPower::abs($difference);

        return NumPower::mean($absolute);
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * ∂L/∂ŷ = sign(ŷ - y)
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

        $difference = NumPower::subtract($output, $target);

        return NumPower::sign($difference);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Mean Absolute Error';
    }
}
