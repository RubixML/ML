<?php

namespace Rubix\Engine\DistanceFunctions;

use Rubix\Engine\Graph\GraphNode;

class Euclidean extends DistanceFunction
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
        return (float) sqrt(array_reduce($this->axes, function ($carry, $axis) use ($a, $b) {
            return $carry += pow($a->get($axis) - $b->get($axis), 2);
        }, 0.0));
    }
}
