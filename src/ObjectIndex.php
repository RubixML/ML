<?php

namespace Rubix\Engine;

use InvalidArgumentException;
use IteratorAggregate;
use RuntimeException;
use ArrayIterator;
use Countable;

class ObjectIndex implements IteratorAggregate, Countable
{
    /**
     * The objects in the index.
     *
     * @var array
     */
    protected $objects = [
        //
    ];

    /**
     * @param  array  $objects
     * @return void
     */
    public function __construct(array $objects = [])
    {
        $this->merge($objects);
    }

    /**
     * Determine if an object exists in the index even if it is null. O(1)
     *
     * @param  mixed  $key
     * @return bool
     */
    public function has($key) : bool
    {
        return array_key_exists($key, $this->objects);
    }

    /**
     * Put an object into the index. O(1)
     *
     * @param  mixed  $key
     * @param  \Rubix\Engine\GraphObject
     * @return self
     */
    public function put($key, GraphObject $object) : ObjectIndex
    {
        if (!is_int($key) && !is_string($key)) {
            throw new InvalidArgumentException('Key must be an integer or string, ' . gettype($key) . ' found.');
        }

        $this->objects[$key] = $object;

        return $this;
    }

    /**
     * Merge an array of objects into the index. O(O)
     *
     * @param  mixed  $objects
     * @return self
     */
    public function merge(array $objects) : ObjectIndex
    {
        $this->objects = array_replace($this->objects, $objects);

        return $this;
    }

    /**
     * Return an object from the index by key or null if not found. O(1)
     *
     * @param  mixed  $key
     * @param  mixed  $default
     * @return mixed
     */
    public function get($key)
    {
        return $this->objects[$key] ?? null;
    }

    /**
     * Return an array of objects from the index with keys intact. O(K)
     *
     * @param  array  $keys
     * @return self
     */
    public function mget(array $keys) : array
    {
        return array_intersect_key($this->objects, array_flip($keys));
    }

    /**
     * Filter the index by a where clause. O(N)
     *
     * @param  string  $property
     * @param  string  $operator
     * @param  mixed  $value
     * @return self
     */
    public function where(string $property, string $operator, $value) : ObjectIndex
    {
        return $this->filter(function ($object) use ($property, $operator, $value) {
            if (!$object->has($property)) {
                return false;
            }

            if ($operator === '===') {
                return $object->get($property) === $value;
            } else if ($operator === '==' || $operator === '=') {
                return $object->get($property) == $value;
            } else if ($operator === '!==') {
                return $object->get($property) !== $value;
            } else if ($operator === '!=' || $operator === '<>') {
                return $object->get($property) != $value;
            } else if ($operator === '>') {
                return $object->get($property) > $value;
            } else if ($operator === '<') {
                return $object->get($property) < $value;
            } else if ($operator === '>=') {
                return $object->get($property) >= $value;
            } else if ($operator === '<=') {
                return $object->get($property) <= $value;
            } else if ($operator === 'like') {
                return strpos($object->get($property), (string) $value) !== false ? true : false;
            }
        });
    }

    /**
     * Return all the objects with property values present in an array of values.
     * O(V*N)
     *
     * @param  string  $property
     * @param  array  $values
     * @return \Rubix\Engine\ObjectIndex
     */
    public function whereIn(string $property, array $values) : ObjectIndex
    {
        return $this->filter(function ($object) use ($property, $values) {
            return in_array($object->get($property), $values);
        });
    }

    /**
     * Order the index by a given object property and direction. O(N)
     *
     * @param  string  $property
     * @param  string  $direction
     * @return self
     */
    public function orderBy(string $property, string $direction = 'ASC') : ObjectIndex
    {
        uasort($this->objects, function ($object1, $object2) use ($property, $direction) {
            if ($object1->get($property) === $object2->get($property)) {
                return 0;
            }

            if ($direction === 'ASC') {
                return ($object1->get($property) < $object2->get($property)) ? -1 : 1;
            } else if ($direction === 'DESC') {
                return ($object1->get($property) > $object2->get($property)) ? -1 : 1;
            } else {
                return 0;
            }
        });

        return $this;
    }

    /**
     * Select multiple columns of property values from an object. O(P*N)
     *
     * @param  mixed  $properties
     * @return array
     */
    public function select($properties) : array
    {
        return array_map(function ($property) {
            return $this->pluck($property);
        }, (array) $properties);
    }

    /**
     * Select a particular column of property values from the index. O(N)
     *
     * @param  string  $property
     * @return array
     */
    public function pluck(string $property) : array
    {
        return array_column($this->objects, $property);
    }

    /**
     * Return the first object in the index. O(1)
     *
     * @return \Rubix\Engine\GraphObject|null
     */
    public function first() : ?GraphObject
    {
        return reset($this->objects) ?: null;
    }

    /**
     * Return the last object in the index. O(1)
     *
     * @return \Rubix\Engine\GraphObject|null
     */
    public function last() : ?GraphObject
    {
        return end($this->objects) ?: null;
    }

    /**
     * Return a random object from the index or null if empty. O(N)
     *
     * @return \Rubix\Engine\GraphObject|null
     */
    public function random() : ?GraphObject
    {
        return $this->get(array_rand($this->objects));
    }

    /**
     * Return all of the objects in the index.
     *
     * @return array
     */
    public function all() : array
    {
        return $this->objects;
    }

    /**
     * Filter the index by the result of a given function.
     *
     * @param  callable  $filter
     * @return self
     */
    public function filter(callable $filter) : ObjectIndex
    {
        return new static(array_filter($this->objects, $filter));
    }

    /**
     * Limit the number of objects in the index.
     *
     * @param  int  $limit
     * @return self
     */
    public function limit(int $limit) : ObjectIndex
    {
        return new static(array_slice($this->objects, 0, $limit, true));
    }

    /**
     * Return a portion of the index after a given offset.
     *
     * @param  int  $offset
     * @return self
     */
    public function skip(int $offset) : ObjectIndex
    {
        return new static(array_slice($this->objects, $offset, null, true));
    }

    /**
     * Get the keys of all the objects in the index.
     *
     * @return array
     */
    public function keys() : array
    {
        return array_keys($this->objects);
    }

    /**
     * Remove an object from the index.
     *
     * @param  mixed  $keys
     * @return self
     */
    public function remove($keys) : ObjectIndex
    {
        foreach ((array) $keys as $key) {
            unset($this->objects[$key]);
        }

        return $this;
    }

    /**
     * Is the index empty?
     *
     * @return bool
     */
    public function isEmpty() : bool
    {
        return empty($this->objects);
    }

    /**
     * Count the number of objects in the index.
     *
     * @return int
     */
    public function count() : int
    {
        return count($this->objects);
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
