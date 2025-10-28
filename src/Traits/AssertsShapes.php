<?php

declare(strict_types=1);

namespace Rubix\ML\Traits;

use InvalidArgumentException;
use NDArray;

/**
 * Validates Shapes
 *
 * A trait that provides shape validation for cost functions to ensure
 * output and target arrays have matching dimensions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
trait AssertsShapes
{
    /**
     * Assert that the output and target NDArrays have identical shapes.
     *
     * @param NDArray $output The output array to check.
     * @param NDArray $target The target array to compare against.
     * @throws InvalidArgumentException If the shapes do not match.
     */
    protected function assertSameShape(NDArray $output, NDArray $target) : void
    {
        if ($output->shape() !== $target->shape()) {
            throw new InvalidArgumentException('Output and target must have identical shapes.');
        }
    }
}
