<?php

namespace Rubix\ML\Graph;

interface Tree
{
    /**
     * Return the root node of the tree.
     *
     * @return \Rubix\ML\Graph\Node|null
     */
    public function root() : ?Node;
}
