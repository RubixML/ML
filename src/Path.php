<?php

namespace Rubix\Graph;

use SplDoublyLinkedList;

class Path extends SplDoublyLinkedList
{
    /**
     * @param  array  $nodes
     * @return void
     */
    public function __construct(array $nodes = [])
    {
        $this->extend($nodes);
    }

    /**
     *  Prepend an node onto the beginning of the path. O(1)
     *
     * @param  \Rubix\Graph\Node  $node
     * @return self
     */
    public function prepend(Node $node) : self
    {
        $this->unshift($node);

        return $this;
    }

    /**
     *  Append a node onto the end of the path. O(1)
     *
     * @param  \Rubix\Graph\Node  $node
     * @return self
     */
    public function append(Node $node) : self
    {
        $this->push($node);

        return $this;
    }

    /**
     * Extend the path with a given array of objects. O(N)
     *
     * @param  array  $nodes
     * @return self
     */
    public function extend(array $nodes) : self
    {
        foreach ($nodes as $node) {
            $this->append($node);
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
     * @return \Rubix\Graph\Node|null
     */
    public function first() : ?Node
    {
        $this->rewind();

        return $this->bottom();
    }

    /**
     * The next object in the path. O(1)
     *
     * @return \Rubix\Graph\Node|null
     */
    public function next() : ?Node
    {
        parent::next();

        return $this->current();
    }

    /**
     * The previous object in the path. O(1)
     *
     * @return \Rubix\Graph\Node|null
     */
    public function prev() : ?Node
    {
        parent::prev();

        return $this->current();
    }

    /**
     * The last object in the path. O(1)
     *
     * @return \Rubix\Graph\Node|null
     */
    public function last() : ?Node
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
