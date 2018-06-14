<?php

namespace Rubix\ML\Graph;

class Edge extends GraphObject
{
    /**
     * The node that this edge connects to.
     *
     * @var  \Rubix\ML\Node
     */
    protected $node;

    /**
     * @param  \Rubix\ML\Node  $node
     * @param  array  $properties
     * @return void
     */
    public function __construct(Node $node, array $properties = [])
    {
        $this->node = $node;

        parent::__construct($properties);
    }

    /**
     * @return \Rubix\ML\Node
     */
    public function node() : Node
    {
        return $this->node;
    }
}
