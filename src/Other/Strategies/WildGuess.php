<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;

/**
 * Wild Guess
 *
 * It is what you think it is. Make a guess somewhere in between the minimum
 * and maximum values observed during fitting with equal probability given to
 * all values within range.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WildGuess implements Continuous
{
    /**
     * The number of decimal places of precision for each guess.
     *
     * @var int
     */
    protected $precision;

    /**
     * The minimum value of the fitted data.
     *
     * @var float|null
     */
    protected $min;

    /**
     * The maximum value of the fitted data.
     *
     * @var float|null
     */
    protected $max;

    /**
     * @param  int  $precision
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $precision = 2)
    {
        if ($precision < 0) {
            throw new InvalidArgumentException('The number of decimal places'
                . ' cannot be less than 0.');
        }

        $this->precision = $precision;
    }

    /**
     * Copy the values.
     *
     * @param  array  $values
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy needs to be fit with'
                . ' at least one value.');
        }

        $this->min = min($values);
        $this->max = max($values);
    }

    /**
     * Choose a random value between the minimum and the maximum of the fitted
     * data.
     *
     * @return mixed
     */
    public function guess()
    {
        if (is_null($this->min) or is_null($this->max)) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        $min = (int) round($this->min * $this->precision);
        $max = (int) round($this->max * $this->precision);

        return rand($min, $max) / $this->precision;
    }
}
