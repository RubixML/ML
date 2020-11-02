<?php

namespace Rubix\ML\Other\Strategies;

/**
 * Strategy
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Strategy
{
    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *
     * @param (string|int|float)[] $values
     */
    public function fit(array $values) : void;

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string;
}
