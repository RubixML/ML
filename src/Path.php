<?php

namespace Rubix\Engine;

use SplDoublyLinkedList;

class Path extends SplDoublyLinkedList
{
    /**
     * @param  array  $objects
     * @return void
     */
    public function __construct(array $objects = [])
    {
        $this->extend($objects);
    }

    /**
     *  Prepend an object onto the beginning of the path. O(1)
     *
     * @param  \Rubix\Engine\GraphObject  $object
     * @return self
     */
    public function prepend(GraphObject $object) : self
    {
        $this->unshift($object);

        return $this;
    }

    /**
     *  Append a node onto the end of the path. O(1)
     *
     * @param  \Rubix\Engine\GraphObject  $object
     * @return self
     */
    public function append(GraphObject $object) : self
    {
        $this->push($object);

        return $this;
    }

    /**
     * Extend the path with a given array of objects. O(N)
     *
     * @param  array  $objects
     * @return self
     */
    public function extend(array $objects) : self
    {
        foreach ($objects as $object) {
            $this->append($object);
        }

        return $this;
    }

    /**
     * Select multiple columns of property values from an object. O(PN)
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
     * Select a particular column of property values. Return default if not found. O(N)
     *
     * @param  string  $property
     * @param  mixed|null  $default
     * @return array
     */
    public function pluck(string $property, $default = null) : array
    {
        $column = [];

        for ($this->rewind(); $this->valid(); $this->next()) {
            $column[] = $this->current()->get($property, $default);
        }

        return $column;
    }

    /**
     * The first object in the path. O(1)
     *
     * @return \Rubix\Engine\GraphObject|null
     */
    public function first() : ?GraphObject
    {
        $this->rewind();

        return $this->bottom();
    }

    /**
     * The next object in the path. O(1)
     *
     * @return \Rubix\Engine\GraphObject|null
     */
    public function next() : ?GraphObject
    {
        parent::next();

        return $this->current();
    }

    /**
     * The previous object in the path. O(1)
     *
     * @return \Rubix\Engine\GraphObject|null
     */
    public function prev() : ?GraphObject
    {
        parent::prev();

        return $this->current();
    }

    /**
     * The last object in the path. O(1)
     *
     * @return \Rubix\Engine\GraphObject|null
     */
    public function last() : ?GraphObject
    {
        return $this->top();
    }

    /**
     * Return an array of all the objects in the path. O(N)
     *
     * @return array
     */
    public function all() : array
    {
        return iterator_to_array($this);
    }

    /**
     * Returns the length of the path in objects.
     *
     * @return int
     */
    public function length() : int
    {
        return $this->count();
    }
}
