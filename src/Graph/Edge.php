<?php

namespace Rubix\Engine\Graph;

class Edge extends GraphObject
{
    /**
     * The node that this edge connects to.
     *
     * @var  \Rubix\Engine\Node
     */
    protected $node;

    /**
     * @param  \Rubix\Engine\Node  $node
     * @param  array  $properties
     * @return void
     */
    public function __construct(Node $node, array $properties = [])
    {
        $this->node = $node;

        parent::__construct($properties);
    }

    /**
     * @return \Rubix\Engine\Node
     */
    public function node() : Node
    {
        return $this->node;
    }
}
