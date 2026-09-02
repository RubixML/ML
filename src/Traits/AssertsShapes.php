<?php

declare(strict_types=1);

namespace Rubix\ML\Traits;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\InvalidFanInException;
use Rubix\ML\Exceptions\InvalidFanOutException;
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
            throw new InvalidArgumentException('Output and target must have the same shape.');
        }
    }

    /**
     * Validating initializer parameters.
     *
     * @param int $fanIn The number of input connections per neuron
     * @param int $fanOut The number of output connections per neuron
     * @throws InvalidFanInException Initializer parameter fanIn is less than 1
     * @throws InvalidFanOutException Initializer parameter fanOut is less than 1
     */
    protected function validateFanInFanOut(int $fanIn, int $fanOut) : void
    {
        if ($fanIn < 1) {
            throw new InvalidFanInException(message: "Fan in cannot be less than 1, $fanIn given");
        }

        if ($fanOut < 1) {
            throw new InvalidFanOutException(message: "Fan out cannot be less than 1, $fanOut given");
        }
    }
}
