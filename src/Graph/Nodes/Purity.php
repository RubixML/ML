<?php

namespace Rubix\ML\Graph\Nodes;

interface Purity extends Node
{
    /**
     * Return the impurity score of the node.
     *
     * @return float
     */
    public function impurity() : float;

    /**
     * Return the number of samples from the training set this node
     * represents.
     *
     * @return int
     */
    public function n() : int;
}
