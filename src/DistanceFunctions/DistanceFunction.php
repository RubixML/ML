<?php

namespace Rubix\Graph\DistanceFunctions;

use Rubix\Graph\GraphNode;

interface DistanceFunction
{
    /**
     * Compute the distance between two nodes.
     *
     * @param  \Rubix\Graph\GraphNode  $start
     * @param  \Rubix\Graph\GraphNode  $end
     * @param  array  $axis
     * @return float
     */
    public function compute(GraphNode $start, GraphNode $end, array $axis) : float;
}
