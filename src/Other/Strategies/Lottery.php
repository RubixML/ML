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
     * The number of categories available to select.
     *
     * @var int|null
     */
    protected $k;

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param array $values
     * @throws \RuntimeException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fit with'
                . ' at least 1 value.');
        }

        $categories = array_values(array_unique($values));

        $this->categories = $categories;
        $this->k = count($categories);
    }

    /**
     * Make a guess.
     *
     * @throws \RuntimeException
     * @return string
     */
    public function guess() : string
    {
        if (!$this->categories or !$this->k) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        $index = rand(0, $this->k - 1);

        return $this->categories[$index];
    }
}
