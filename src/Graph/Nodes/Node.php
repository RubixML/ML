<?php

namespace Rubix\ML\Graph\Nodes;

use Generator;

interface Node
{
    /**
     * Return a generator for all of the node's edges i.e. the nodes that
     * this node connects to.
     *
     * @return Generator
     */
    public function edges() : Generator;
}
