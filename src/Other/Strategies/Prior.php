<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;
use Stringable;

use function count;

/**
 * Prior
 *
 * Make a guess based on the class prior probability.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Prior implements Categorical, Stringable
{
    /**
     * The counts of each unique class.
     *
     * @var int[]|null
     */
    protected $counts;

    /**
     * The sample size.
     *
     * @var int|null
     */
    protected $n;

    /**
     * Fit the guessing strategy to a set of values.
     * @internal
     *
     * @param string[] $values
     * @throws \InvalidArgumentException
     */
    public function fit(array $values) : void
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Strategy must be fitted'
                . ' to at least 1 value.');
        }

        $this->counts = array_count_values($values);
        $this->n = count($values);
    }

    /**
     * Make a categorical guess.
     *
     * @internal
     *
     * @throws \RuntimeException
     * @return string
     */
    public function guess() : string
    {
        if (!$this->counts or !$this->n) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        $r = rand(0, $this->n);

        foreach ($this->counts as $class => $count) {
            $r -= $count;

            if ($r <= 0) {
                return (string) $class;
            }
        }

        return (string) key($this->counts);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Prior';
    }
}
