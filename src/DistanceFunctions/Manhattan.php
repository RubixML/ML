<?php

namespace Rubix\Graph\DistanceFunctions;

use Rubix\Graph\GraphNode;

class Manhattan implements DistanceFunction
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
        return (float) array_reduce($axis, function ($carry, $axis) use ($start, $end) {
            return $carry += abs($start->get($axis) - $end->get($axis));
        }, 0.0);
    }
}
