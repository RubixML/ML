<?php

namespace Rubix\Engine\DistanceFunctions;

use Rubix\Engine\GraphNode;

class Euclidean implements DistanceFunction
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
        return (float) sqrt(array_reduce($axis, function ($carry, $axis) use ($start, $end) {
            return $carry += pow($start->get($axis) - $end->get($axis), 2);
        }, 0.0));
    }
}
