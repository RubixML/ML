<?php

namespace Rubix\ML\Graph;

use InvalidArgumentException;

class GraphNode extends GraphObject implements Node
{
    /**
     * The identifier of the node.
     *
     * @var mixed
     */
    protected $id;

    /**
     * An index of edges.
     *
     * @var \Rubix\ML\ObjectIndex
     */
    protected $edges;

    /**
     * An autoincrementing counter that keeps track of issued node IDs.
     *
     * @var int
     */
    protected static $counter = 1;

    /**
     * Reset the autoincrementing counter to a particular offset value.
     *
     * @param  int  $offset
     * @return void
     */
    public static function resetCounter(int $offset = 1) : void
    {
        if ($offset < 0) {
            throw new InvalidArgumentException('Autoincrementing counter offset must be non-negative.');
        }

        self::$counter = $offset;
    }

    /**
     * @param  array  $properties
     * @return void
     */
    public function __construct(array $properties = [])
    {
        $this->id = self::$counter++;
        $this->edges = new ObjectIndex();

        parent::__construct($properties);
    }

    /**
     * @return mixed
     */
    public function id()
    {
        return $this->id;
    }

    /**
     * @return array
     */
    public function edges() : ObjectIndex
    {
        return $this->edges;
    }

    /**
     * Attach an edge linking this node to another node.
     *
     * @param  \Rubix\ML\GraphNode  $node
     * @param  array  $properties
     * @return \Rubix\ML\Edge
     */
    public function attach(GraphNode $node, array $properties = []) : Edge
    {
        $edge = new Edge($node, $properties);

        $this->edges->put($node->id(), $edge);

        return $edge;
    }

    /**
     * Remove an edge from the node.
     *
     * @param  \Rubix\ML\GraphNode  $node
     * @return void
     */
    public function detach(GraphNode $node) : self
    {
        $this->edges->remove($node->id());

        return $this;
    }

    /**
     * Check if this node is the same as a given node.
     *
     * @param  \Rubix\ML\GraphNode  $node
     * @return bool
     */
    public function isSame(GraphNode $node) : bool
    {
        return $this->id() === $node->id();
    }

    /**
     * Is the node a leaf node? i.e having no children.
     *
     * @return bool
     */
    public function isLeaf() : bool
    {
        return $this->edges->isEmpty();
    }
}
