<?php

namespace Rubix\ML;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use IteratorAggregate;
use ArrayAccess;
use Traversable;
use Countable;

use function count;
use function func_get_args;

/**
 * Tuple
 *
 * An immutable list with a fixed-length that is indexable by offset in the sequence.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements ArrayAccess<int, mixed>
 * @implements IteratorAggregate<int, mixed>
 */
class Tuple implements ArrayAccess, IteratorAggregate, Countable
{
    /**
     * The elements of the tuple.
     *
     * @var list<mixed>
     */
    protected array $elements;

    public function __construct()
    {
        $this->elements = func_get_args();
    }

    /**
     * List the elements of the tuple in an array.
     *
     * @return list<mixed>
     */
    public function list() : array
    {
        return $this->elements;
    }

    /**
     * Return the number of elements in the tuple.
     *
     * @return int
     */
    public function count() : int
    {
        return count($this->elements);
    }

    /**
     * Return a row from the dataset at the given offset.
     *
     * @param int $offset
     * @throws InvalidArgumentException
     * @return mixed
     */
    #[\ReturnTypeWillChange]
    public function offsetGet($offset)
    {
        if (isset($this->elements[$offset])) {
            return $this->elements[$offset];
        }

        throw new InvalidArgumentException("Element at offset $offset not found.");
    }

    /**
     * @param int $offset
     * @param mixed[] $values
     * @throws RuntimeException
     */
    public function offsetSet($offset, $values) : void
    {
        throw new RuntimeException('Tuples cannot be mutated.');
    }

    /**
     * Does a given row exist in the dataset.
     *
     * @param int $offset
     * @return bool
     */
    public function offsetExists($offset) : bool
    {
        return isset($this->elements[$offset]);
    }

    /**
     * @param int $offset
     * @throws RuntimeException
     */
    public function offsetUnset($offset) : void
    {
        throw new RuntimeException('Tuples cannot be mutated.');
    }

    /**
     * Get an iterator for the elements in the tuple.
     *
     * @return \Generator<mixed>
     */
    public function getIterator() : Traversable
    {
        yield from $this->elements;
    }
}
