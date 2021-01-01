<?php

namespace Rubix\ML\Other\Strategies;

use Stringable;

/**
 * Strategy
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Strategy extends Stringable
{
    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *
     * @param (string|int|float)[] $values
     */
    public function fit(array $values) : void;
}
