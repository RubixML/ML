<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;

/**
 * Lottery
 *
 * Hold a lottery in which each category has an equal chance of being picked.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Lottery implements Categorical
{
    /**
     * The possible categories.
     *
     * @var array
     */
    protected $categories = [
        //
    ];

    /**
     * The number of categories in the lottery.
     *
     * @var int
     */
    protected $n;

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param  array  $values
     * @throws \RuntimeException
     * @return void
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy needs to be fit with'
                . ' at least one value.');
        }

        $categories = array_values(array_unique($values));

        $this->categories = $categories;
        $this->n = count($categories);
    }

    /**
     * Make a categorical guess.
     *
     * @throws \RuntimeException
     * @return string
     */
    public function guess() : string
    {
        if (empty($this->categories)) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->categories[rand(0, $this->n - 1)];
    }
}
