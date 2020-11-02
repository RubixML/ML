<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;
use Stringable;

use const Rubix\ML\PHI;

/**
 * Wild Guess
 *
 * Guess a random number somewhere between the minimum and maximum computed by fitting a
 * collection of values.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WildGuess implements Continuous, Stringable
{
    /**
     * The minimum value of the fitted data.
     *
     * @var int|null
     */
    protected $min;

    /**
     * The maximum value of the fitted data.
     *
     * @var int|null
     */
    protected $max;

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *
     * @param (int|float)[] $values
     * @throws \InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fitted'
                . ' to at least 1 value.');
        }

        $this->min = (int) round(min($values) * PHI);
        $this->max = (int) round(max($values) * PHI);
    }

    /**
     * Make a continuous guess.
     *
     * @internal
     *
     * @throws \RuntimeException
     * @return float
     */
    public function guess() : float
    {
        if ($this->min === null or $this->max === null) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return rand($this->min, $this->max) / PHI;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Wild Guess';
    }
}
