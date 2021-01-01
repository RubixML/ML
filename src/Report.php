<?php

namespace Rubix\ML;

use ArrayAccess;
use Countable;
use Generator;
use IteratorAggregate;
use JsonSerializable;
use Stringable;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Other\Helpers\JSON;

/**
 * Report
 *
 * The results of a cross-validation report.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements ArrayAccess<int, array>
 * @implements IteratorAggregate<int, array>
 */
class Report implements ArrayAccess, JsonSerializable, IteratorAggregate, Stringable, Countable
{
    /**
     * The attributes that make up the report.
     *
     * @var mixed[]
     */
    protected $attributes;

    /**
     * @param mixed[] $attributes
     */
    public function __construct(array $attributes)
    {
        $this->attributes = $attributes;
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
     * Return a JSON representation of the report.
     *
     * @param bool $pretty
     * @return \Rubix\ML\Encoding
     */
    public function toJSON(bool $pretty = true) : Encoding
    {
        return new Encoding(JSON::encode($this, $pretty ? JSON_PRETTY_PRINT : 0) ?: '');
    }

    /**
     * @param string|int $key
     * @param mixed[] $values
     * @throws \Rubix\ML\Exceptions\RuntimeException
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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return mixed
     */
    public function offsetGet($key)
    {
        if (isset($this->attributes[$key])) {
            return $this->attributes[$key];
        }

        throw new InvalidArgumentException("Attribute with key $key not found.");
    }

    /**
     * @param string|int $key
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function offsetUnset($key) : void
    {
        throw new RuntimeException('Reports cannot be mutated directly.');
    }

    /**
     * @return mixed[]
     */
    public function jsonSerialize() : array
    {
        return $this->toArray();
    }

    /**
     * Get an iterator for the attributes in the report.
     *
     * @return \Generator<mixed>
     */
    public function getIterator() : Generator
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
     * Return a human-readable string representation of the report.
     *
     * @return string
     */
    public function __toString() : string
    {
        return (string) $this->toJSON(true) . PHP_EOL;
    }
}
