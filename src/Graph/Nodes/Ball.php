<?php

namespace Rubix\ML\Graph\Nodes;

interface Ball extends Node
{
    /**
     * Return the center vector.
     *
     * @return (int|float)[]
     */
    public function center() : array;

    /**
     * Return the radius of the centroid.
     *
     * @return float
     */
    public function radius() : float;
}
