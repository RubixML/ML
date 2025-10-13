<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\CostFunctions\LeastSquares\Base\Contracts;

use NDArray;
use Stringable;

/**
 * Cost Function
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface CostFunction extends Stringable
{
    /**
     * Compute the loss score.
     *
     * @internal
     *
     * @param NDArray $output
     * @param NDArray $target
     * @return float
     */
    public function compute(NDArray $output, NDArray $target) : float;

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @internal
     *
     * @param NDArray $output
     * @param NDArray $target
     * @return NDArray
     */
    public function differentiate(NDArray $output, NDArray $target) : NDArray;
}
