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
class Constant implements Continuous
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
     * @param (int|float)[] $values
     */
    public function fit(array $values) : void
    {
        //
    }

    /**
     * Make a guess.
     *
     * @return int|float
     */
    public function guess()
    {
        return $this->value;
    }
}
