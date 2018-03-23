<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

use Rubix\Engine\Graph\GraphNode;

class Manhattan extends DistanceFunction
{
    /**
     * Compute the distance between two nodes.
     *
     * @param  \Rubix\Engine\GraphNode  $a
     * @param  \Rubix\Engine\GraphNode  $b
     * @return float
     */
    public function compute(GraphNode $a, GraphNode $b) : float
    {
        return (float) array_reduce($this->axes, function ($carry, $axis) use ($a, $b) {
            return $carry += abs($a->get($axis) - $b->get($axis));
        }, 0.0);
    }
}
