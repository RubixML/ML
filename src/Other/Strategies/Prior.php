<?php

namespace Rubix\ML\Other\Strategies;

use InvalidArgumentException;
use RuntimeException;

/**
 * Prior
 *
 * Make a guess based on the class prior probability.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Prior implements Categorical
{
    /**
     * The counts of each unique class.
     *
     * @var array
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
     *
     * @param array $values
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
     * Return the prior probabilities of each class.
     *
     * @return float[]
     */
    public function priors() : array
    {
        $priors = [];

        foreach ($this->counts as $class => $count) {
            $priors[$class] = $count / $this->n;
        }

        return $priors;
    }

    /**
     * Make a categorical guess.
     *
     * @throws \RuntimeException
     * @return string
     */
    public function guess() : string
    {
        if ($this->n === null or !$this->counts) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        $class = current($this->counts);

        $r = rand(0, $this->n);

        foreach ($this->counts as $class => $count) {
            $r -= $count;

            if ($r <= 0) {
                break 1;
            }
        }

        return (string) $class;
    }
}
