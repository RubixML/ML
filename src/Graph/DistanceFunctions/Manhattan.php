<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

use Rubix\Engine\Graph\GraphNode;
use InvalidArgumentException;

class Manhattan extends DistanceFunction
{
    /**
     * Measure the distance between two nodes.
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

        $distance = 0.0;

        foreach ($a as $i => $coordinate) {
            $distance += abs($coordinate - $b[$i]);
        }

        return (float) $distance;
    }
}
