<?php

namespace Rubix\Engine\Math;

use IteratorAggregate;
use ArrayIterator;
use Countable;

class Set implements IteratorAggregate, Countable
{
    /**
     * The items in the set.
     *
     * @var array
     */
    protected $values;

    /**
     * @param  array  $values
     * @return void
     */
    public function __construct(array $values = [])
    {
        $values = array_values(array_unique($values));

        sort($values);

        $this->values = $values;
    }

    /**
     * Returns the cardinality of the set.
     *
     * @return int
     */
    public function cardinality() : int
    {
        return count($this->values);
    }

    /**
     * @return array
     */
    public function values() : array
    {
        return $this->values;
    }

    /**
     * Return the union of this set and another set.
     *
     * @param  \Rubix\Engine\Set  $set
     * @return self
     */
    public function union(self $set) : self
    {
        return new static(array_merge($this->values, $set->values()));
    }

    /**
     * Return the intersection of this set and another set.
     *
     * @param  \Rubix\Engine\Set  $set
     * @return self
     */
    public function intersection(self $set) : self
    {
        return new static(array_intersect($this->values, $set->values()));
    }

    /**
     * Calculate the difference between this set and another set.
     *
     * @param  \Rubix\Engine\Set  $set
     * @return self
     */
    public function difference(self $set) : self
    {
        return new static(array_diff($this->values, $set->values()));
    }

    /**
     * Calculate the cartesian product of this set and a given set. i.e. all
     * ordered pairs.
     *
     * @param  \Rubix\Engine\Set  $set
     * @return array
     */
    public function product(self $set) : array
    {
        $product = [];

        foreach ($this->values as $value) {
            foreach ($set->values() as $multiplicand) {
                $product[] = new static(array_merge([$multiplicand], [$value]));
            }
        }

        return $product;
    }

    /**
     * Build the power set of this set. i.e. an array containing all subsets of
     * the original set. O(2^N)
     *
     * @return array
     */
    public function power() : array
    {
        $power = [new static()];

        foreach ($this->values as $value) {
            foreach ($power as $multiplier) {
                $power[] = new static(array_merge([$value], $multiplier->values()));
            }
        }

        return $power;
    }

    /**
     * Alias of cardinality().
     *
     * @return int
     */
    public function count() : int
    {
        return $this->cardinality();
    }

    /**
     * Is the set empty?
     *
     * @return bool
     */
    public function isEmpty() : bool
    {
        return $this->cardinality() <= 0;
    }

    /**
     * Get an iterator for the objects in the index.
     *
     * @return \ArrayIterator
     */
    public function getIterator()
    {
        return new ArrayIterator($this->objects);
    }
}
