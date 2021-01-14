<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\DataType;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Mean
 *
 * This strategy always predicts the mean of the fitted data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Mean implements Strategy
{
    /**
     * The mean of the fitted values.
     *
     * @var float|null
     */
    protected $mean;

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
        return isset($this->mean);
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *  @param list<int|float> $values
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be'
                . ' fitted to at least 1 value.');
        }

        $this->mean = Stats::mean($values);
    }

    /**
     * Make a continuous guess.
     *
     * @internal
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return float
     */
    public function guess() : float
    {
        if ($this->mean === null) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->mean;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Mean';
    }
}
