<?php

namespace Rubix\ML\Strategies;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function array_slice;

/**
 * K Most Frequent
 *
 * This Strategy outputs one of k most frequently occurring classes at random with equal probability.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KMostFrequent implements Strategy
{
    /**
     * The number of most frequent classes to consider.
     *
     * @var int
     */
    protected int $k;

    /**
     * The k most frequent classes.
     *
     * @var list<string>
     */
    protected array $classes = [
        //
    ];

    /**
     * @param int $k
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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
     * Return the data type the strategy handles.
     *
     * @return \Rubix\ML\DataType
     */
    public function type() : DataType
    {
        return DataType::categorical();
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
        return !empty($this->classes);
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *
     * @param list<string> $values
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be'
                . ' fitted to at least 1 value.');
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
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return string
     */
    public function guess() : string
    {
        if (!$this->classes) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        $offset = array_rand($this->classes);

        return $this->classes[$offset];
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "K Most Frequent (k: {$this->k})";
    }
}
