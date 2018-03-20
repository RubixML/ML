<?php

namespace Rubix\Graph\DistanceFunctions;

use Rubix\Graph\GraphNode;

class Minkowski implements DistanceFunction
{
    /**
     * @param  float  $lambda
     * @return void
     */
    public function __construct(float $lambda = 3.0)
    {
        $this->lambda = $lambda;
    }

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
        return (float) pow(array_reduce($axis, function ($carry, $axis) use ($start, $end) {
            return $carry += pow(abs($start->get($axis) - $end->get($axis)), $this->lambda);
        }, 0.0), 1.0 / $this->lambda);
    }
}
