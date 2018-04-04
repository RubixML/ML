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
     * Compute the distance between two coordinate vectors.
     *
     * @param  \Rubix\Engine\GraphNode  $a
     * @param  \Rubix\Engine\GraphNode  $b
     * @return float
     */
    public function compute(GraphNode $a, GraphNode $b) : float
    {
        return (float) pow(array_reduce($this->axes, function ($distance, $axis) use ($a, $b) {
            return $distance += pow(abs($a->get($axis) - $b->get($axis)), $this->lambda);
        }, 0.0), 1.0 / $this->lambda);
    }

    /**
     * Compute the distance given two coordinate vectors.
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
            $distance += pow(abs($coordinate - $b[$i]), $this->lambda);
        }

        return (float) pow($distance, 1.0 / $this->lambda);
    }
}
