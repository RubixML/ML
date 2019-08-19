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
     * @var float
     */
    protected $value;

    /**
     * @param float $value
     */
    public function __construct(float $value = 0.)
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
