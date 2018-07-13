<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;

class WildGuess implements Continuous
{
    /**
     * The number of decimal precision of the guess.
     *
     * @var int
     */
    protected $decimals;

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
     * @param  int  $decimals
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $decimals = 2)
    {
        if ($decimals < 0) {
            throw new InvalidArgumentException('The number of decimals must be'
                . ' 0 or greater.');
        }

        $this->decimals = $decimals;
    }

    /**
     * Return the range of possible guesses for this strategy in a tuple.
     *
     * @return array
     */
    public function range() : array
    {
        return [$this->min, $this->max];
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

        if ($this->decimals === 0) {
            return rand((int) $this->min, (int) $this->max);
        }

        return rand((int) ($this->min * $this->decimals),
            (int) ($this->max * $this->decimals)) / $this->decimals;
    }
}
