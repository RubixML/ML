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
trait ValidatesShapes
{
    /**
     * Validate that output and target have the same shape.
     *
     * @param NDArray $output
     * @param NDArray $target
     * @throws InvalidArgumentException
     */
    protected function validateShapes(NDArray $output, NDArray $target) : void
    {
        if ($output->shape() !== $target->shape()) {
            throw new InvalidArgumentException('Output and target must have the same shape.');
        }
    }
}
