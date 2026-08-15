<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions;

use NDArray;
use NumPower;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Traits\AssertsShapes;
use const Rubix\ML\EPSILON;

/**
 * Cross Entropy
 *
 * Cross Entropy, or log loss, measures the performance of a classification model
 * whose output is a probability value between 0 and 1. Cross-entropy loss
 * increases as the predicted probability diverges from the actual label. So
 * predicting a probability of .012 when the actual observation label is 1 would
 * be bad and result in a high loss value. A perfect score would have a log loss
 * of 0.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class CrossEntropy implements ClassificationLoss
{
    use AssertsShapes;

    public function __construct()
    {
        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();
    }

    /**
     * Compute the loss score.
     *
     * L(y, ŷ) = -Σ(y * log(ŷ)) / n
     *
     * @param NDArray $output The output of the network
     * @param NDArray $target The target values
     * @return float
     */
    public function compute(NDArray $output, NDArray $target) : float
    {
        $this->assertSameShape($output, $target);

        // Clip values to avoid log(0)
        $output = NumPower::clip($output, EPSILON, 1.0);

        $logOutput = NumPower::log($output);
        $product = NumPower::multiply($target, $logOutput);
        $negated = NumPower::multiply($product, -1.0);

        return NumPower::mean($negated);
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * ∂L/∂ŷ = (ŷ - y) / (ŷ * (1 - ŷ))
     *
     * @param NDArray $output The output of the network
     * @param NDArray $target The target values
     * @return NDArray
     */
    public function differentiate(NDArray $output, NDArray $target) : NDArray
    {
        $this->assertSameShape($output, $target);

        // Numerator = ŷ - y (calculate before clipping to preserve zeros)
        $numerator = NumPower::subtract($output, $target);

        // Clip values to avoid division by zero
        $output = NumPower::clip($output, EPSILON, 1.0 - EPSILON);

        // Denominator = ŷ * (1 - ŷ)
        $oneMinusOutput = NumPower::subtract(1.0, $output);
        $denominator = NumPower::multiply($output, $oneMinusOutput);
        $denominator = NumPower::clip($denominator, EPSILON, 1.0);

        return NumPower::divide($numerator, $denominator);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Cross Entropy';
    }
}
