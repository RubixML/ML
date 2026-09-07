<?php

namespace Rubix\ML;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use IteratorAggregate;
use ArrayAccess;
use Traversable;
use Countable;
use JsonSerializable;

use function count;
use function implode;

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
class Tuple implements ArrayAccess, IteratorAggregate, Countable, JsonSerializable
{
    /**
     * The elements of the tuple.
     *
     * @var list<mixed>
     */
    protected array $elements;

    /**
     * @param mixed ...$elements
     */
    public function __construct(mixed ...$elements)
    {
        $this->elements = $elements;
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
     * Return the element at the given offset.
     *
     * @param int $offset
     * @throws InvalidArgumentException
     * @return mixed
     */
    public function offsetGet(mixed $offset) : mixed
    {
        if ($offset < 0 or $offset >= $this->count()) {
            throw new InvalidArgumentException("Element at offset $offset not found.");
        }

        return $this->elements[$offset];
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
     * Does an element exist at the given offset.
     *
     * @param int $offset
     * @return bool
     */
    public function offsetExists(mixed $offset) : bool
    {
        return $offset >= 0 && $offset < $this->count();
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

    /**
     * Return the elements of the tuple as a JSON-serializable array.
     *
     * @return list<mixed>
     */
    public function jsonSerialize() : array
    {
        return $this->elements;
    }

    /**
     * Return the string representation of the tuple.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return '(' . implode(', ', $this->elements) . ')';
    }
}
