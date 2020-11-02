<?php

namespace Rubix\ML\Graph\Nodes;

use Traversable;

/**
 * Hypercube
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Hypercube extends Node
{
    /**
     * Return the minimum bounding box surrounding this node.
     *
     * @return \Traversable<array>
     */
    public function sides() : Traversable;

    /**
     * Does the hypercube reduce to a single point?
     *
     * @return bool
     */
    public function isPoint() : bool;
}
