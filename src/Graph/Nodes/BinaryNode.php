<?php

namespace Rubix\ML\Graph\Nodes;

/**
 * Binary Node
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface BinaryNode extends Node
{
    /**
     * Return the height of the node in the tree.
     *
     * @return int
     */
    public function height() : int;
}
