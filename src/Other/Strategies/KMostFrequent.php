<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;

/**
 * K Most Frequent
 *
 * This Strategy outputs one of K most frequently occurring classes at random.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KMostFrequent extends Lottery
{
    /**
     * The number of most frequent categories to guess from.
     *
     * @var int
     */
    protected $k;

    /**
     * @param  int  $k
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 1)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Cannot guess less than 1'
                . ' category.');
        }

        $this->k = $k;
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @param  array  $values
     * @throws \InvalidArgumentException;
     * @return void
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fit with'
                . ' at least 1 value.');
        }

        $categories = array_count_values($values);

        arsort($categories);

        $categories = array_slice($categories, 0, $this->k, true);

        $this->categories = array_keys($categories);
    }
}
