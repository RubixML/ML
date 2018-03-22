<?php

namespace Rubix\Engine\DistanceFunctions;

use Rubix\Engine\Graph\GraphNode;

interface DistanceFunction
{
    /**
     * Compute the distance between two nodes.
     *
     * @param  \Rubix\Engine\GraphNode  $start
     * @param  \Rubix\Engine\GraphNode  $end
     * @param  array  $axis
     * @return float
     */
    public function compute(GraphNode $start, GraphNode $end, array $axis) : float;
}
