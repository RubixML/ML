<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Graph\Nodes\Leaf;

interface BST extends BinaryTree
{
    /**
     * Search the tree for a leaf node or return null if not found.
     *
     * @param (string|int|float)[] $sample
     * @return \Rubix\ML\Graph\Nodes\Leaf|null
     */
    public function search(array $sample);
}
