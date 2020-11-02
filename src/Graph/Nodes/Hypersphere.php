<?php

namespace Rubix\ML\Graph\Nodes;

/**
 * Hypersphere
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Hypersphere extends Node
{
    /**
     * Return the centroid of the hypersphere.
     *
     * @return list<string|int|float>
     */
    public function center() : array;

    /**
     * Return the radius of the centroid.
     *
     * @return float
     */
    public function radius() : float;

    /**
     * Does the hypersphere reduce to a single point?
     *
     * @return bool
     */
    public function isPoint() : bool;
}
