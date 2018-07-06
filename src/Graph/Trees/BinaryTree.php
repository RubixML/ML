<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\BinaryNode;

class BinaryTree implements Tree
{
    /**
     * The root node of the tree.
     *
     * @var \Rubix\ML\Graph\Nodes\Node|null
     */
    protected $root;


    /**
     * @return \Rubix\ML\Graph\Nodes\Node|null
     */
    public function root() : ?Node
    {
        return $this->root;
    }

    /**
     * The height of the tree. O(V) because node heights are not memoized.
     *
     * @return int
     */
    public function height() : int
    {
        return isset($this->root) ? $this->root->height() : 0;
    }

    /**
     * The balance factor of the tree. O(V) because balance requires height of
     * each node.
     *
     * @return int
     */
    public function balance() : int
    {
        return isset($this->root) ? $this->root->balance() : 0;
    }

    /**
     * Is the tree bare?
     *
     * @return bool
     */
    public function bare() : bool
    {
        return !isset($this->root);
    }
}
