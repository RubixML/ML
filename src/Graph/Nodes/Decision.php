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
interface Decision extends BinaryNode
{
    /**
     * Return the impurity of the labels as a result of the decision.
     *
     * @return float
     */
    public function impurity() : float;

    /**
     * Return the number of samples that are represented in the subtree stemming from this node.
     *
     * @return int
     */
    public function n() : int;
}
