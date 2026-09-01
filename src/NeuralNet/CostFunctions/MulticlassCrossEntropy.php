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
 * Multiclass Cross Entropy
 *
 * Multiclass Cross Entropy measures the performance of a multiclass
 * classification model whose output is a probability distribution over the
 * possible classes. Cross-entropy loss increases as the predicted probability
 * distribution diverges from the actual distribution.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MulticlassCrossEntropy implements ClassificationLoss
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
     * @param NDArray $output
     * @param NDArray $target
     * @return float
     */
    public function compute(NDArray $output, NDArray $target) : float
    {
        $this->assertSameShape($output, $target);

        $output = NumPower::clip($output, EPSILON, 1.0);

        $logOutput = NumPower::log($output);

        $product = NumPower::multiply($target, $logOutput);

        $negated = NumPower::multiply($product, -1.0);

        return NumPower::mean($negated);
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * ∂L/∂ŷ = -y / ŷ
     *
     * @param NDArray $output
     * @param NDArray $target
     * @return NDArray
     */
    public function differentiate(NDArray $output, NDArray $target) : NDArray
    {
        $this->assertSameShape($output, $target);

        $output = NumPower::clip($output, EPSILON, 1.0);

        $negated = NumPower::multiply($target, -1.0);

        return NumPower::divide($negated, $output);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Multiclass Cross Entropy';
    }
}
