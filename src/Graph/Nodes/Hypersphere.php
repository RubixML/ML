<?php

namespace Rubix\ML\Graph\Nodes;

interface Hypersphere extends Node
{
    /**
     * Return the centroid of the hypersphere.
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
