<?php

namespace Rubix\Graph\DistanceFunctions;

use Rubix\Graph\GraphNode;

class Diagonal implements DistanceFunction
{
    /**
     * Compute the distance between two nodes.
     *
     * @param  \Rubix\Graph\GraphNode  $start
     * @param  \Rubix\Graph\GraphNode  $end
     * @param  array  $axis
     * @return float
     */
    public function compute(GraphNode $start, GraphNode $end, array $axis) : float
    {
        return (float) max(array_map(function ($axis) use ($start, $end) {
            return abs($start->get($axis) - $end->get($axis));
        }, $axis));
    }
}
