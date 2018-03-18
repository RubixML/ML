<?php

namespace Rubix\Graph;

class Edge extends GraphObject
{
    /**
     * The node that this edge connects to.
     *
     * @var  \Rubix\Graph\Node
     */
    protected $node;

    /**
     * @param  \Rubix\Graph\Node  $node
     * @param  array  $properties
     * @return void
     */
    public function __construct(Node $node, array $properties = [])
    {
        $this->node = $node;

        parent::__construct($properties);
    }

    /**
     * @return \Rubix\Graph\Node
     */
    public function node() : Node
    {
        return $this->node;
    }
}
