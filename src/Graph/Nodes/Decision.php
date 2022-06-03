<?php

namespace Rubix\ML\Graph\Nodes;

/**
 * Decision
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Decision
{
    /**
     * Return the impurity of the labels within the node.
     *
     * @return float
     */
    public function impurity() : float;

    /**
     * Return the number of samples that are represented in the subtree stemming from this node.
     *
     * @return int<0,max>
     */
    public function n() : int;
}
