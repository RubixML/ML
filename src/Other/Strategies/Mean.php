<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;
use RuntimeException;

/**
 * Mean
 *
 * This strategy always predicts the mean of the fitted data.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Mean implements Continuous
{
    /**
     * The mean of the fitted values.
     *
     * @var float|null
     */
    protected $mean;

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param (string|int|float)[] $values
     * @throws \InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fitted'
                . ' to at least 1 value.');
        }

        $this->mean = Stats::mean($values);
    }

    /**
     * Make a continuous guess.
     *
     * @throws \RuntimeException
     * @return float
     */
    public function guess() : float
    {
        if ($this->mean === null) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->mean;
    }
}
