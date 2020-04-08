<?php

namespace Rubix\ML\Graph\Nodes;

interface Hypercube extends Node
{
    /**
     * Return the minimum bounding box surrounding this node.
     *
     * @return iterable<array>
     */
    public function sides() : iterable;
}
