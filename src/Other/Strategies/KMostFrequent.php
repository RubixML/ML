<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;

class KMostFrequent implements Categorical
{
    /**
     * The number of most frequent categorical variables to guess from.
     *
     * @var int
     */
    protected $k;

    /**
     * The k most frequent categories to consider when formulating a guess.
     *
     * @var array
     */
    protected $categories = [
        //
    ];

    /**
     * @param  int  $k
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 1)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Cannot make a guess amongst'
                . ' less than 1 category.');
        }

        $this->k = $k;
    }

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
     * Rank the classes by most frequent and chose the top k.
     *
     * @param  array  $values
     * @throws \InvalidArgumentException;
     * @return void
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy needs to be fit with'
                . ' at least one value.');
        }

        $categories = array_count_values($values);

        arsort($categories);

        $this->categories = array_keys(array_slice($categories, 0, $this->k));
    }

    /**
     * Select a random discrete value amongst the top k.
     *
     * @throws \RuntimeException
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
