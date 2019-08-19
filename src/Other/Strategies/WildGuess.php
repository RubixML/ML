<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\PHI;

/**
 * Wild Guess
 *
 * Guess a random number somewhere between an upper and lower bound given by
 * the data and a user-defined *shrinkage* parameter.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WildGuess implements Continuous
{
    /**
     * The range between the upper and lower bounds of the guess. A value of 1.0
     * indicates the full range of fitted values, whereas the range becomes narrower
     * as the parameter goes to 0.
     */
    protected $alpha;

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
     * @param float $alpha
     * @throws \InvalidArgumentException
     */
    public function __construct(float $alpha = 0.5)
    {
        if ($alpha <= 0. or $alpha > 1.) {
            throw new InvalidArgumentException('Alpha must be between'
                . " 0 and 1, $alpha given.");
        }

        $this->alpha = $alpha;
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
            throw new InvalidArgumentException('Strategy must be fitted'
                . ' to at least 1 value.');
        }

        $this->min = $this->alpha * min($values);
        $this->max = $this->alpha * max($values);
    }

    /**
     * Return the lower and upper bounds in a 2-tuple.
     *
     * @return (int|float|null)[]
     */
    public function range() : array
    {
        return [$this->min, $this->max];
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

        $min = (int) round($this->min * PHI);
        $max = (int) round($this->max * PHI);

        return rand($min, $max) / PHI;
    }
}
