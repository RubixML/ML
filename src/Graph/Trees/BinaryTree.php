<?php

namespace Rubix\ML\Graph\Trees;

interface BinaryTree extends Tree
{
    /**
     * Return the height of the tree.
     *
     * @return int
     */
    public function height() : int;

    /**
     * Return the balance of the tree.
     *
     * @return int
     */
    public function balance() : int;
}
