<?php

namespace Rubix\Graph;

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
     * @var \Rubix\Graph\ObjectIndex
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
     * @param  \Rubix\Graph\GraphNode  $node
     * @param  array  $properties
     * @return \Rubix\Graph\Edge
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
     * @param  \Rubix\Graph\GraphNode  $node
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
     * @param  \Rubix\Graph\GraphNode  $node
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
