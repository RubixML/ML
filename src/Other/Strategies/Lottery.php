<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;

class Lottery implements Categorical
{
    /**
     * The unique outcomes each having equal chance of winning lottery.
     *
     * @var array
     */
    protected $categories = [
        //
    ];

    /**
     * Return the set of all possible guesses for this strategy in an array.
     *
     * @return array
     */
    public function set() : array
    {
        return $this->categories;
    }

    /**
     * Store every unique outcome.
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

        $this->categories = array_values(array_unique($values));
    }

    /**
     * Hold a lottery in which each category has an equal chance of being picked.
     *
     * @return mixed
     */
    public function guess()
    {
        if (empty($this->categories)) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->categories[array_rand($this->categories)];
    }
}
