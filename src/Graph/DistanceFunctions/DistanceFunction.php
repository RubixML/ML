<?php

namespace Rubix\Engine\Graph\DistanceFunctions;

use Rubix\Engine\Graph\GraphNode;

abstract class DistanceFunction
{
    /**
     * The property names that represent the axes of the grid.
     *
     * @var array
     */
    protected $axes;

    /**
     * @param  array  $axes
     * @return self
     */
    public function __construct(array $axes = ['x', 'y'])
    {
        $this->axes = $axes;
    }

    /**
     * The number of dimensions in the grid space.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return count($this->axes);
    }

    /**
     * @return array
     */
    public function axes() : array
    {
        return $this->axes;
    }

    /**
     * Measure the distance between two nodes.
     *
     * @param  \Rubix\Engine\GraphNode  $a
     * @param  \Rubix\Engine\GraphNode  $b
     * @return float
     */
    abstract public function compute(GraphNode $a, GraphNode $b) : float;

    /**
     * Compute the distance given two coordinate vectors.
     *
     * @param  array  $a
     * @param  array  $b
     * @return float
     */
    abstract public function distance(array $a, array $b) : float;
}
