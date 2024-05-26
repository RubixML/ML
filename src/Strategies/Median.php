<?php

namespace Rubix\ML\Strategies;

use Rubix\ML\DataType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Median
 *
 * This strategy always predicts the median of the fitted data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Median implements Strategy
{
    /**
     * The median of the fitted values.
     *
     * @var float|null
     */
    protected ?float $median = null;

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
        return isset($this->median);
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     * @param list<int|float> $values
     * @throws InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be'
                . ' fitted to at least 1 value.');
        }

        $this->median = Stats::median($values);
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
        if ($this->median === null) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->median;
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
        return 'Median';
    }
}
