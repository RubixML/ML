<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

use Rubix\Engine\Graph\GraphNode;

class Minkowski extends DistanceFunction
{
    /**
     * @param  array  $axes
     * @param  float  $lambda
     * @return void
     */
    public function __construct(array $axes = ['x', 'y'], float $lambda = 3.0)
    {
        $this->lambda = $lambda;

        parent::__construct($axes);
    }

    /**
     * Compute the distance between node a and b.
     *
     * @param  \Rubix\Engine\GraphNode  $a
     * @param  \Rubix\Engine\GraphNode  $b
     * @return float
     */
    public function compute(GraphNode $a, GraphNode $b) : float
    {
        return (float) pow(array_reduce($this->axes, function ($carry, $axis) use ($a, $b) {
            return $carry += pow(abs($a->get($axis) - $b->get($axis)), $this->lambda);
        }, 0.0), 1.0 / $this->lambda);
    }
}
