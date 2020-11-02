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
     * Return the impurity score as a result of the decision.
     *
     * @return float
     */
    public function impurity() : float;

    /**
     * Return the number of samples from the training set that this node represents.
     *
     * @return int
     */
    public function n() : int;
}
