<?php

namespace Rubix\ML\Graph\Trees;

interface BinaryTree extends Tree
{
    /**
     * Return the balance factor of the tree. A balanced tree will have
     * a factor of 0 whereas an imbalanced tree will either be positive
     * or negative indicating the direction and degree of the imbalance.
     *
     * @return int
     */
    public function balance() : int;
}
