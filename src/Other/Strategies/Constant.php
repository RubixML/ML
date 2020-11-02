<?php

namespace Rubix\ML\Other\Strategies;

use Stringable;

/**
 * Constant
 *
 * Always guess the same value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Constant implements Continuous, Stringable
{
    /**
     * The value to constantly guess.
     *
     * @var int|float
     */
    protected $value;

    /**
     * @param int|float $value
     */
    public function __construct($value = 0)
    {
        $this->value = $value;
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *
     * @param (int|float)[] $values
     */
    public function fit(array $values) : void
    {
        //
    }

    /**
     * Make a guess.
     *
     * @internal
     *
     * @return int|float
     */
    public function guess()
    {
        return $this->value;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Constant (value: {$this->value})";
    }
}
