<?php

namespace Rubix\ML\Graph\Nodes;

interface Spatial extends Node
{
    /**
     * Return the bounding box around this node.
     * 
     * @return array[]
     */
    public function box() : array;
}