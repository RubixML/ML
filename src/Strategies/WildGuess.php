<?php

namespace Rubix\ML\Strategies;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Wild Guess
 *
 * Guess a random number somewhere between the minimum and maximum computed by fitting a collection of values.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WildGuess implements Strategy
{
    /**
     * The minimum value of the fitted data.
     *
     * @var int|null
     */
    protected ?int $min = null;

    /**
     * The maximum value of the fitted data.
     *
     * @var int|null
     */
    protected ?int $max = null;

    /**
     * A constant that determines the precision of the floating point numbers.
     *
     * @var float|null
     */
    protected ?float $phi = null;

    /**
     * Return the data type the strategy handles.
     *
     * @return DataType
     */
    public function type() : DataType
    {
        return DataType::continuous();
    }

    /**
     * Has the strategy been fitted?
     *
     * @internal
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->min, $this->max, $this->phi);
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *
     * @param (int|float)[] $values
     * @throws InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fitted'
                . ' to at least 1 value.');
        }

        $min = min($values);
        $max = max($values);

        $phi = getrandmax() / max(abs($max), abs($min));

        $this->min = (int) floor($min * $phi);
        $this->max = (int) ceil($max * $phi);
        $this->phi = $phi;
    }

    /**
     * Make a continuous guess.
     *
     * @internal
     *
     * @throws RuntimeException
     * @return float
     */
    public function guess() : float
    {
        if ($this->min === null or $this->max === null or $this->phi === null) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return rand($this->min, $this->max) / $this->phi;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Wild Guess';
    }
}
