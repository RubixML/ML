<?php

namespace Rubix\Engine;

class Node extends GraphObject
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
     * @var \Rubix\Engine\ObjectIndex
     */
    protected $edges;

    /**
     * @param  mixed  $id
     * @param  array  $properties
     * @return void
     */
    public function __construct($id, array $properties = [])
    {
        $this->id = $id;
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
     * @param  \Rubix\Engine\Node  $node
     * @param  string  $relationship
     * @param  array  $properties
     * @return \Rubix\Engine\Edge
     */
    public function attach(Node $node, array $properties = []) : Edge
    {
        $edge = new Edge($node, $properties);

        $this->edges->put($node->id(), $edge);

        return $edge;
    }

    /**
     * Remove an edge from the node.
     *
     * @param  \Rubix\Engine\Node  $node
     * @param  string  $relationship
     * @return void
     */
    public function detach(Node $node) : void
    {
        $this->edges->remove($node->id());
    }

    /**
     * Check if this node is the same as a given node.
     *
     * @param  \Rubix\Engine\Node  $node
     * @return bool
     */
    public function isSame(Node $node) : bool
    {
        return $this->id() === $node->id();
    }

    /**
     * Is the node a leaf node? I.e having no children.
     *
     * @return bool
     */
    public function isLeaf() : bool
    {
        return $this->edges->isEmpty();
    }
}
