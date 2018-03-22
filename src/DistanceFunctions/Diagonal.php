<?php

namespace Rubix\Engine\DistanceFunctions;

use Rubix\Engine\GraphNode;

class Diagonal implements DistanceFunction
{
    /**
     * Compute the distance between two nodes.
     *
     * @param  \Rubix\Engine\GraphNode  $start
     * @param  \Rubix\Engine\GraphNode  $end
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
