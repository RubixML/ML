<?php

namespace Rubix\ML\Graph\Nodes;

interface BoundingBox extends Node
{
    /**
     * Return the bounding box surrounding this node.
     *
     * @return array[]
     */
    public function box() : array;
}
