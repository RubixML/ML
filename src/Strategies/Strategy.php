<?php

namespace Rubix\ML\Strategies;

use Rubix\ML\DataType;
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
     * Return the data type the strategy handles.
     *
     * @return \Rubix\ML\DataType
     */
    public function type() : DataType;

    /**
     * Has the strategy been fitted?
     *
     * @internal
     *
     * @return bool
     */
    public function fitted() : bool;

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *
     * @param list<string|int|float> $values
     */
    public function fit(array $values) : void;

    /**
     * Make a guess.
     *
     * @internal
     *
     * @return string|int|float
     */
    public function guess();
}
