<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

use Rubix\Engine\Graph\GraphNode;
use InvalidArgumentException;

class Diagonal extends DistanceFunction
{
    /**
     * Measure the distance between node a and b.
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

    /**
     * Compute the distance between two coordinate vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @throws \InvalidArgumentException
     * @return float
     */
    public function distance(array $a, array $b) : float
    {
        if (count($a) !== count($b)) {
            throw new InvalidArgumentException('The size of each coordinate vector must be equal.');
        }

        return (float) max(array_map(function ($ca, $cb) {
            return abs($ca - $cb);
        }, $a, $b));
    }
}
