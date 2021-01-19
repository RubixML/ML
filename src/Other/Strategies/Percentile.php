<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\DataType;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Percentile
 *
 * A strategy that always guesses the p-th percentile of the fitted data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Percentile implements Strategy
{
    /**
     * The percentile of the fitted data to use as a guess.
     *
     * @var float
     */
    protected $p;

    /**
     * The pth percentile of the fitted data.
     *
     * @var float|null
     */
    protected $percentile;

    /**
     * @param float $p
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $p = 50.0)
    {
        if ($p < 0.0 or $p > 100.0) {
            throw new InvalidArgumentException('Percentile must be'
                . " between 0 and 100, $p given.");
        }

        $this->p = $p;
    }

    /**
     * Return the data type the strategy handles.
     *
     * @return \Rubix\ML\DataType
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
        return isset($this->percentile);
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *
     * @param list<int|float> $values
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fitted'
                . ' to at least 1 value.');
        }

        $q = $this->p / 100.0;

        $this->percentile = Stats::quantile($values, $q);
    }

    /**
     * Make a guess.
     *
     * @internal
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return float
     */
    public function guess() : float
    {
        if ($this->percentile === null) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->percentile;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Percentile (p: {$this->p})";
    }
}
