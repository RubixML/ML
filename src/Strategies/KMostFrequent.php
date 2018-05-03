<?php

namespace Rubix\Engine\Strategies;

use InvalidArgumentException;

class KMostFrequent implements Categorical
{
    /**
     * The number of most frequent classes to guess from.
     *
     * @var int
     */
    protected $k;

    /**
     * The k most frequent classes to consider when imputing data.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param  int  $k
     * @return void
     */
    public function __construct($k = 1)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('The value of k cannot be less than 1.');
        }

        $this->k = $k;
    }

    /**
     * Rank the classes by most frequent and chose the top k.
     *
     * @param  array  $values
     * @return mixed
     */
    public function fit(array $values) : void
    {
        $classes = array_count_values($values);

        arsort($classes);

        $this->classes = array_keys(array_slice($classes, 0, $this->k));
    }

    /**
     * Impute a missing value by selecting a random class among the top k.
     *
     * @return mixed
     */
    public function guess()
    {
        return $this->classes[array_rand($this->classes)];
    }
}
