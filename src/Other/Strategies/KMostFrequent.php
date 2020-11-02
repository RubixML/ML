<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;
use Stringable;

use function array_slice;

/**
 * K Most Frequent
 *
 * This Strategy outputs one of k most frequently occurring classes at random
 * with equal probability.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KMostFrequent implements Categorical, Stringable
{
    /**
     * The number of most frequent classes to consider.
     *
     * @var int
     */
    protected $k;

    /**
     * The k most frequent classes.
     *
     * @var string[]
     */
    protected $classes = [
        //
    ];

    /**
     * @param int $k
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 1)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Cannot guess from'
                . " less than 1 class, $k given.");
        }

        $this->k = $k;
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *
     * @param (string|int)[] $values
     * @throws \InvalidArgumentException;
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fitted'
                . ' to at least 1 value.');
        }

        $classes = array_count_values($values);

        arsort($classes);

        $classes = array_slice($classes, 0, $this->k, true);

        $this->classes = array_keys($classes);
    }

    /**
     * Make a guess.
     *
     * @internal
     *
     * @throws \RuntimeException
     * @return string
     */
    public function guess() : string
    {
        if (!$this->classes) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        return $this->classes[rand(0, $this->k - 1)];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "K Most Frequent (k: {$this->k})";
    }
}
