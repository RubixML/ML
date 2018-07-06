<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Graph\Nodes\Node;

interface Tree
{
    /**
     * Return the root node of the tree.
     *
     * @return \Rubix\ML\Graph\Nodes\Node|null
     */
    public function root() : ?Node;
}
