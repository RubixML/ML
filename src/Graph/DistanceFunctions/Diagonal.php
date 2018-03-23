<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

use Rubix\Engine\Graph\GraphNode;

class Diagonal extends DistanceFunction
{
    /**
     * Compute the distance between node a and b.
     *
     * @param  \Rubix\Engine\GraphNode  $a
     * @param  \Rubix\Engine\GraphNode  $b
     * @return float
     */
    public function compute(GraphNode $a, GraphNode $b) : float
    {
        return (float) max(array_map(function ($axis) use ($a, $b) {
            return abs($a->get($axis) - $b->get($axis));
        }, $this->axes));
    }
}
