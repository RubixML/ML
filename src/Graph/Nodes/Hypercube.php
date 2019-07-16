<?php

namespace Rubix\ML\Graph\Nodes;

use Generator;

interface Hypercube extends Node
{
    /**
     * Return the minimum bounding box surrounding this node.
     *
     * @return \Generator
     */
    public function sides() : Generator;
}
