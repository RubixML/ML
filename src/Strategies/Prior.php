<?php

namespace Rubix\ML\Strategies;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

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
class Prior implements Strategy
{
    /**
     * The counts of each unique class.
     *
     * @var int[]|null
     */
    protected ?array $counts = null;

    /**
     * The sample size.
     *
     * @var int|null
     */
    protected ?int $n = null;

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
        return isset($this->counts);
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
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return string
     */
    public function guess() : string
    {
        if (!$this->counts or !$this->n) {
            throw new RuntimeException('Strategy has not been fitted.');
        }

        $r = rand(0, $this->n);

        /** @var string $class */
        foreach ($this->counts as $class => $count) {
            $r -= $count;

            if ($r <= 0) {
                return $class;
            }
        }

        return (string) key($this->counts);
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
        return 'Prior';
    }
}
