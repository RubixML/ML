<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions;

use NDArray;
use NumPower;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Traits\AssertsShapes;

/**
 * Least Squares
 *
 * Least Squares or *quadratic* loss is a function that measures the squared
 * error between the target output and the actual output of a network.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class LeastSquares implements RegressionLoss
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
     * L(y, ŷ) = Σ(y - ŷ)^2 / n
     *
     * @param NDArray $output
     * @param NDArray $target
     * @return float
     */
    public function compute(NDArray $output, NDArray $target) : float
    {
        $this->assertSameShape($output, $target);

        $difference = NumPower::subtract($output, $target);
        $squared = NumPower::pow($difference, 2);

        return NumPower::mean($squared);
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * ∂L/∂ŷ = y - ŷ
     *
     * @param NDArray $output
     * @param NDArray $target
     * @return NDArray
     */
    public function differentiate(NDArray $output, NDArray $target) : NDArray
    {
        $this->assertSameShape($output, $target);

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
