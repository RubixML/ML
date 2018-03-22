<?php

namespace Rubix\Engine\DistanceFunctions;

use Rubix\Engine\GraphNode;

class Manhattan implements DistanceFunction
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
        return (float) array_reduce($axis, function ($carry, $axis) use ($start, $end) {
            return $carry += abs($start->get($axis) - $end->get($axis));
        }, 0.0);
    }
}
