<?php

namespace Rubix\Engine\Preprocessors\Strategies;

use InvalidArgumentException;

class KMostFrequent implements Categorical
{
    /**
     * The k most frequent classes to consider when making a guess.
     *
     * @var int
     */
    protected $k;

    /**
     * @param  int  $k
     * @return void
     */
    public function __construct($k = 3)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('The value of k cannot be less than 1.');
        }

        $this->k = $k;
    }

    /**
     * Pick one of the top values at random.
     *
     * @param  array  $values
     * @return mixed
     */
    public function guess(array $values)
    {
        $classes = array_count_values($values);

        arsort($classes);

        $classes = array_slice($classes, 0, $this->k);

        return array_rand($classes);
    }
}
