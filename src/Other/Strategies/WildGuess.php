<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\Other\Helpers\Stats;
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
     * @param int $precision
     * @throws \InvalidArgumentException
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
     * Fit the guessing strategy to a set of values.
     *
     * @param array $values
     * @throws \InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fit with'
                . ' at least 1 value.');
        }

        [$this->min, $this->max] = Stats::range($values);
    }

    /**
     * Make a continuous guess.
     *
     * @throws \RuntimeException
     * @return float
     */
    public function guess() : float
    {
        if ($this->min === null or $this->max === null) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        $min = (int) round($this->min * $this->precision);
        $max = (int) round($this->max * $this->precision);

        return rand($min, $max) / $this->precision;
    }
}
