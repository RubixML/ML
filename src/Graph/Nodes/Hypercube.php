<?php

namespace Rubix\ML\Graph\Nodes;

use Traversable;

interface Hypercube extends Node
{
    /**
     * Return the minimum bounding box surrounding this node.
     *
     * @return \Traversable<array>
     */
    public function sides() : Traversable;
}
