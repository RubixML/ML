<?php

declare(strict_types=1);

namespace Rubix\ML;

use Rubix\ML\Exceptions\InvalidArgumentException;
use ArrayAccess;
use IteratorAggregate;
use Traversable;
use Countable;

use function count;
use function array_keys;
use function array_key_exists;
use function array_walk;

/**
 * Set
 *
 * An unordered collection of unique, scalar values backed by an associative
 * array in which every member maps to null.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements ArrayAccess<int|string, bool>
 * @implements IteratorAggregate<int, int|string>
 */
class Set implements ArrayAccess, IteratorAggregate, Countable
{
    /**
     * The members of the set keyed by their value. Every value is null and is
     * used only to mark the presence of a member in the underlying hash.
     *
     * @var array<int|string, null>
     */
    protected array $members = [];

    /**
     * Build a new set from the given members.
     *
     * A single array or Traversable may be provided to seed the set with its
     * contents, or any number of individual members may be provided.
     *
     * @param mixed ...$members
     */
    public function __construct(mixed ...$members)
    {
        array_walk($members, [$this, 'add']);
    }

    /**
     * Add a member to the set.
     *
     * @param string|int $member
     */
    public function add(string|int $member) : void
    {
        $this->members[$member] = null;
    }

    /**
     * Remove a member from the set.
     *
     * @param string|int $member
     */
    public function remove(string|int $member) : void
    {
        unset($this->members[$member]);
    }

    /**
     * Check whether the given member is present in the set.
     *
     * @param string|int $member
     * @return bool
     */
    public function has(string|int $member) : bool
    {
        return array_key_exists($member, $this->members);
    }

    /**
     * List the members of the set as a plain array.
     *
     * @return list<int|string>
     */
    public function toArray() : array
    {
        return array_keys($this->members);
    }

    /**
     * Return the number of members in the set.
     *
     * @return int
     */
    public function count() : int
    {
        return count($this->members);
    }

    /**
     * Is the given member present in the set.
     *
     * @param mixed $offset
     * @return bool
     */
    public function offsetExists($offset) : bool
    {
        return $this->has($offset);
    }

    /**
     * Add the given member to the set.
     *
     * Members are stored as null, so the supplied value is ignored and only
     * the offset is considered.
     *
     * @param mixed $offset
     * @param mixed $values
     * @throws InvalidArgumentException
     */
    public function offsetSet($offset, $values) : void
    {
        if (!is_int($offset) and !is_string($offset)) {
            throw new InvalidArgumentException('Set members must be of type int or string.');
        }

        $this->add($offset);
    }

    /**
     * Check whether the given member is present in the set.
     *
     * @param mixed $offset
     * @throws InvalidArgumentException
     * @return bool
     */
    #[\ReturnTypeWillChange]
    public function offsetGet($offset)
    {
        if (!is_int($offset) and !is_string($offset)) {
            throw new InvalidArgumentException('Set members must be of type int or string.');
        }

        if (!$this->has($offset)) {
            throw new InvalidArgumentException('Member not found in the set.');
        }

        return true;
    }

    /**
     * Remove the given member from the set.
     *
     * @param mixed $offset
     * @throws InvalidArgumentException
     */
    public function offsetUnset($offset) : void
    {
        if (!is_int($offset) and !is_string($offset)) {
            throw new InvalidArgumentException('Set members must be of type int or string.');
        }

        $this->remove($offset);
    }

    /**
     * Get an iterator for the members of the set.
     *
     * @return \Generator<int|string, mixed>
     */
    public function getIterator() : Traversable
    {
        yield from array_keys($this->members);
    }
}
