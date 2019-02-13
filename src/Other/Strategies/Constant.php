<?php

namespace Rubix\ML\Other\Strategies;

/**
 * Constant
 *
 * Always guess a constant value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Constant implements Continuous
{
    /**
     * The value to guess.
     *
     * @var float
     */
    protected $value;

    /**
     * @param float $value
     */
    public function __construct(float $value)
    {
        $this->value = $value;
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param array $values
     * @throws \InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        //
    }

    /**
     * Make a continuous guess.
     *
     * @return float
     */
    public function guess() : float
    {
        return $this->value;
    }
}
