<?php

namespace Rubix\ML\Other\Strategies;

/**
 * Constant
 *
 * Always guess the same value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Constant implements Categorical, Continuous
{
    /**
     * The value to constantly guess.
     *
     * @var string|int|float
     */
    protected $value;

    /**
     * @param string|int|float $value
     */
    public function __construct($value = 0)
    {
        $this->value = $value;
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param (string|int|float)[] $values
     */
    public function fit(array $values) : void
    {
        //
    }

    /**
     * Make a continuous guess.
     *
     * @return string|int|float
     */
    public function guess()
    {
        return $this->value;
    }
}
