<?php

namespace Rubix\ML;

use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use IteratorAggregate;
use JsonSerializable;
use ArrayAccess;
use Traversable;
use Stringable;
use Countable;

/**
 * Report
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements ArrayAccess<int, array>
 * @implements IteratorAggregate<int, array>
 */
class Report implements ArrayAccess, JsonSerializable, IteratorAggregate, Countable, Stringable
{
    /**
     * The attributes that make up the report.
     *
     * @var mixed[]
     */
    protected array $attributes;

    /**
     * @param mixed[] $attributes
     */
    public function __construct(array $attributes)
    {
        $this->attributes = $attributes;
    }

    /**
     * Return a JSON representation of the report.
     *
     * @param bool $pretty
     * @return Encoding
     */
    public function toJSON(bool $pretty = true) : Encoding
    {
        $options = $pretty ? JSON_PRETTY_PRINT : 0;

        return new Encoding(JSON::encode($this, $options));
    }

    /**
     * Return an array representation of the report.
     *
     * @return mixed[]
     */
    public function toArray() : array
    {
        return $this->attributes;
    }

    /**
     * @param string|int $key
     * @param mixed[] $values
     * @throws RuntimeException
     */
    public function offsetSet($key, $values) : void
    {
        throw new RuntimeException('Reports cannot be mutated directly.');
    }

    /**
     * Does a given row exist in the dataset.
     *
     * @param string|int $key
     * @return bool
     */
    public function offsetExists($key) : bool
    {
        return isset($this->attributes[$key]);
    }

    /**
     * Return an attribute from the report with the given key.
     *
     * @param string|int $key
     * @throws InvalidArgumentException
     * @return mixed
     */
    #[\ReturnTypeWillChange]
    public function offsetGet($key)
    {
        if (isset($this->attributes[$key])) {
            return $this->attributes[$key];
        }

        throw new InvalidArgumentException("Attribute with key $key not found.");
    }

    /**
     * @param string|int $key
     * @throws RuntimeException
     */
    public function offsetUnset($key) : void
    {
        throw new RuntimeException('Reports cannot be mutated directly.');
    }

    /**
     * Get an iterator for the attributes in the report.
     *
     * @return \Generator<mixed>
     */
    public function getIterator() : Traversable
    {
        yield from $this->attributes;
    }

    /**
     * Return the number of level 1 attributes in the report.
     *
     * @return int
     */
    public function count() : int
    {
        return count($this->attributes);
    }

    /**
     * @return mixed[]
     */
    public function jsonSerialize() : array
    {
        return $this->toArray();
    }

    /**
     * Return a human-readable string representation of the report.
     *
     * @return string
     */
    public function __toString() : string
    {
        return (string) $this->toJSON(true) . PHP_EOL;
    }
}
